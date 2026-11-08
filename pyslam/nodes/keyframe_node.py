from typing import Tuple
from pynodes import Node
import time
import os
import shutil

import numpy as np
import cv2
from PIL import Image

from pyslam.preprocessing import vggto as vggto_preprocess
from pyslam.video import TimedVideoPublisher
from pyslam.keyframes import FrameTracker
from pyslam import conf as slam_conf


class KeyFrameNode(Node):
    name="video"

    def __init__(self, node_id, inbox, outbound, stop_event, *, video_src_path:str=slam_conf.VIDEO_SRC_PATH):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.video_src_path = video_src_path
        self.tgt_aspect_ratio: float|None = None
        self.tgt_size: Tuple[int]|None = None

    def setup(self):
        self.advertise("/video/chunks")
        self.create_kf_directory('./keyframes')

    def create_kf_directory(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def spin(self):
        self.setup()

        count = 0
        video_producer = TimedVideoPublisher(self.video_src_path, fps=30)
        kf_filtering = FrameTracker()
        kf_count = 0

        chunk_kfids = []
        img_chunk = []

        while not self.stop_event.is_set():
            img = video_producer.read()
            if img is None:
                self.logger.info(f"Publishing chunk with kfids: {chunk_kfids}")
                self.publish(
                    "/video/chunks",
                    {
                        "kf_ids": chunk_kfids,
                        "imgs": img_chunk,
                    }
                )
                self.publish(
                    "/video/chunks",
                    {
                        "done": True
                    }
                )
                break

            count += 1
            img = self.process_img(img)

            enough_disparity = kf_filtering.compute_disparity(img, slam_conf.MIN_DISPARITY)
            if not enough_disparity:
                continue

            kf_id = kf_count
            kf_count += 1

            chunk_kfids.append(kf_id)
            img_chunk.append(img)

            self.save_image(kf_id, img)

            if len(chunk_kfids) == slam_conf.CHUNK_SIZE:
                self.logger.info(f"Publishing chunk with kfids: {chunk_kfids}")
                self.publish(
                    "/video/chunks",
                    {
                        "kf_ids": chunk_kfids,
                        "imgs": img_chunk,
                    }
                )

                chunk_kfids = chunk_kfids[-slam_conf.CHUNK_OVERLAP:]
                img_chunk = img_chunk[-slam_conf.CHUNK_OVERLAP:]

        video_producer.release()

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
        Image.fromarray(img).save(f"./keyframes/{kfid}.png")
