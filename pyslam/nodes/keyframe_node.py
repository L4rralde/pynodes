from typing import Tuple, List
from pynodes import Node
import time
import os
import shutil
import queue
import threading

import numpy as np
import cv2
from PIL import Image

from pyslam.preprocessing import vggto as vggto_preprocess
from pyslam.video import TimedVideoPublisher
from pyslam.keyframes import FrameTracker
from pyslam import conf as slam_conf
from pyslam.utilities.shmem import AllocatedShmImage


class KeyFrameNode(Node):
    name="video"

    def __init__(self, node_id, inbox, outbound, stop_event, *, video_src_path:str=slam_conf.VIDEO_SRC_PATH, fps: int = slam_conf.FPS):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.video_src_path = video_src_path
        self.tgt_aspect_ratio: float|None = None
        self.tgt_size: Tuple[int]|None = None
        self.shm_message = AllocatedShmImage("keyframes", max_num_bytes=512*512*4*slam_conf.CHUNK_SIZE)
        self.fps = fps

    def setup(self):
        self.advertise("/video/chunks")
        self.create_kf_directory('./keyframes_dir')

    def create_kf_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def video_feader(self, frame_q: queue.Queue) -> None:
        count = 0
        video_producer = TimedVideoPublisher(
            self.video_src_path,
            fps=self.fps
        )
        kf_filtering = FrameTracker()
        try:
            while not self.stop_event.is_set():
                img = video_producer.read()
                if img is None:
                    break

                img = self.process_img(img)
                
                enough_disparity = kf_filtering.compute_disparity(
                    img,
                    slam_conf.MIN_DISPARITY
                )

                frame_q.put({
                    'frame_data':{
                        'img': img,
                        'enough_disparity': enough_disparity,
                        'id': count
                    }
                })
                count += 1
        finally:
            frame_q.put({'done': True})
            self.logger.info("Video finished")
            video_producer.release()
        
    def spin(self):
        self.setup()

        kf_count = 0
        chunk_cnt = 0

        chunk_kfids = []
        img_chunk = []

        frame_q = queue.Queue()

        producer_thread = threading.Thread(
            target=self.video_feader,
            args=(frame_q,),
            daemon=True
        )
        time.sleep(20)
        producer_thread.start()

        try:
            while True:
                try:
                    video_info_msg = frame_q.get(timeout=0.05)
                except queue.Empty:
                    if not producer_thread.is_alive():
                        break
                    continue
                try:
                    if video_info_msg.get('done', False):
                        self.logger.info("Received done")
                        if len(chunk_kfids) > slam_conf.CHUNK_OVERLAP:
                            self.publish_chunk(
                                img_chunk,
                                chunk_kfids,
                                chunk_cnt
                            )
                        break

                    frame_info = video_info_msg['frame_data']
                    frame_cnt = frame_info['id']
                    img = frame_info['img']
                    enough_disparity = frame_info['enough_disparity']

                    cv2.imshow("video", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                    if not enough_disparity:
                        continue

                    kf_id = kf_count
                    kf_count += 1
                    self.save_image(kf_id, img)
                    chunk_kfids.append(kf_id)
                    img_chunk.append(img)

                    if len(chunk_kfids) == slam_conf.CHUNK_SIZE:
                        self.publish_chunk(
                            img_chunk,
                            chunk_kfids,
                            chunk_cnt
                        )

                        chunk_kfids = chunk_kfids[-slam_conf.CHUNK_OVERLAP:]
                        img_chunk = img_chunk[-slam_conf.CHUNK_OVERLAP:]
                        chunk_cnt += 1

                finally:
                    frame_q.task_done()

        except Exception:
            self.logger.exception("Error while processing video frames")

        finally:
            cv2.destroyWindow("video")
            producer_thread.join()
            frame_q.join()
            self.finish()

    def process_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.tgt_aspect_ratio is None or self.tgt_size is None:
            self.tgt_aspect_ratio = h/w
            if not 0.5 <= self.tgt_aspect_ratio <= 2.0:
                raise RuntimeError(f"Invalid aspect ratio: {self.tgt_aspect_ratio}")

            self.tgt_size = vggto_preprocess.get_target_size(img, mode='max_size')

        aspect_ratio = h/w
        if abs(aspect_ratio - self.tgt_aspect_ratio) > 1e-3:
            raise RuntimeError("Received image with different aspect ratio")

        target_h, target_w = self.tgt_size        
        img = cv2.resize(img, (target_w, target_h), cv2.INTER_CUBIC)

        return img

    def save_image(self, kfid: int, img: np.ndarray) -> None:
        Image.fromarray(img).save(f"./keyframes_dir/{kfid}.png")

    def publish_chunk(
            self,
            img_list: List[np.ndarray],
            kfids: List[int],
            chunk_cnt: int
        ) -> None:
        img_array = np.asarray(img_list)
        self.shm_message.fill(img_array)
        self.logger.info(f"Publishing chunk (id: {chunk_cnt}) with kfids: {kfids}")
        self.publish(
            "/video/chunks",
            {
                "shm": True,
                "kf_ids": kfids,
                "imgs": self.shm_message.get_metadata(),
                "chunk_cnt": chunk_cnt
            }
        )

    def finish(self) -> None:
        self.publish(
            "/video/chunks",
            {
                "done": True
            }
        )
        self.logger.info("Done sent")
        self.shm_message.release()
