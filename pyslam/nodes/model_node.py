from typing import Dict, List, Tuple

import torch
import torchvision.transforms.v2 as T
import numpy as np

from pynodes import Node, Message
import pyslam.conf as slam_conf
from vggt_salad_slam.transforms.utils import to_ndarrays_for_pointcloud
from vggt_salad_slam.dtypes import Prediction
from pyslam.utilities.kf_storage import KeyFrameStorage
from pyslam.utilities.allocate_data import AllocatedTorchImgChunk
from pyslam.utilities.shmem import(
    AllocatedShmImage,
    AllocatedShmPrediction,
    AllocatedSharedMemoryNdarray
)
from pyslam.model.model import VeggietSaladOmega


#FIXME: This node is not tested and some bugs are known.
class FoundationModelNode(Node):
    name = "3dgfm"
    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.kf_storage = KeyFrameStorage()
        self.input_chunk: AllocatedTorchImgChunk = AllocatedTorchImgChunk(slam_conf.CHUNK_SIZE)
        self.closing_chunk: AllocatedTorchImgChunk = AllocatedTorchImgChunk(2*slam_conf.CHUNK_OVERLAP)
        self.model: VeggietSaladOmega = VeggietSaladOmega()
        self.chunk_cnt: int=0
        self.prev_kfids: List[int] = []
        self.shm_prediction_o = AllocatedShmPrediction()

    def setup(self):
        self.subscribe("/video/chunks", self.on_input)
        self.subscribe("/closed_loop/closing_chunk", self.on_closed_loop)
        self.advertise("/3dgfm/predictions")
        self.advertise("/vpr/descriptors")
        self.logger.info("Loading model...")
        self.model.load_model(slam_conf.VPR_REPO_PATH, slam_conf.VGGT_OMEGA_CKPT)
        self.logger.info("Model loaded")

    def finish(self) -> None:
        if self.done:
            return
        self.logger.info("Finishing")
        self.done = True
        self.publish(
            "/3dgfm/predictions",
            {
                "done": True
            }
        )

    def to_torch_img(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1)/255.0

    def append_chunks(self, kfids: List[int], imgs: List[torch.Tensor]) -> None:
        for kfid, img in zip(kfids, imgs):
            if kfid in self.prev_kfids:
                continue
            self.kf_storage.add(kfid, img)

    def get_msg_imgs(self, msg: Message) -> np.ndarray:
        shm = msg.payload.get('shm', False)
        if not shm:
            return (msg.payload['imgs'], None)

        buff, imgs = AllocatedSharedMemoryNdarray.attach(msg.payload['imgs'])
        return buff, imgs

    def publish_predictions(self, chunk_id: int, kfids: List[int], preds: Prediction) -> None:
        self.shm_prediction_o.fill(preds)
        self.publish(
            "/3dgfm/predictions",
            {
                "type": "measurement",
                "chunk_id": chunk_id,
                "kf_ids": kfids,
                "predictions": self.shm_prediction_o.get_metadata(),
                "shm": True
            }
        )

    def on_input(self, msg: Message) -> None:
        if 'done' in msg.payload:
            self.finish()
            return

        kfids = msg.payload['kf_ids']
        buff, imgs = self.get_msg_imgs(msg)
        self.logger.info(f"Received chunk with kfids: {kfids}")

        imgs = [
            self.to_torch_img(img)
            for img in imgs
        ]

        buff.close()

        if not self.input_chunk.is_initialized():
            self.input_chunk.set_img_shape(imgs[0].shape[-2:])
            self.closing_chunk.set_img_shape(imgs[0].shape[-2:])

        self.input_chunk.fill(imgs)
        x = self.input_chunk.get()

        self.logger.info("Calling forward pass")
        preds = self.model.model_prediction(x, kfids)
        self.logger.info("Finished forward pass")

        chunk_id = self.chunk_cnt
        self.prev_kfids = kfids
        self.chunk_cnt += 1

        descriptors = preds.pop('descriptor')
        self.logger.info(f"Publishing descriptors with id: {chunk_id}")
        self.publish(
            "/vpr/descriptors",
            {
                "chunk_id": chunk_id,
                "kf_ids": kfids,
                "descriptors": descriptors
            }
        )
        preds = Prediction.from_dict(preds)

        self.logger.info(f"Publishing predictions with id: {chunk_id}")
        self.publish_predictions(
            chunk_id,
            kfids,
            preds
        )
    
    def on_closed_loop(self, msg: Message) ->  None:
        pass


class VprNode(FoundationModelNode):
    name="vpr"
    def setup(self):
        super().setup()
        self.model = torch.hub.load(
            "gmberton/MegaLoc",
            "get_trained_model"
        ).eval()

        self.transform =  T.Compose([
            #T.ToTensor(), #The input is already in the correct format
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(size=[322, 322], antialias=True),
        ])

    def on_chunk(self, msg):
        if 'done' in msg.payload:
            self.done = True
            self.logger.info("Finishing")
            self.publish(
                "/vpr/descriptors",
                {
                    'done': True
                }
            )
            return
        kfids = msg.payload['kf_ids']
        imgs = msg.payload['imgs']
        self.logger.info(f"Received chunk with kfids: {kfids}")

        imgs = [
            torch.from_numpy(img).permute(2, 0, 1)/255.0 #torch mdoels expect normalized [0,1] images
            for img in imgs
        ]

        for kfid, img in zip(kfids, imgs):
            if kfid in self.prev_kfids:
                continue
            self.kf_storage.add(kfid, img)

        h, w = imgs[0].shape[-2:]
        if self.input_chunk is None:
            self.input_chunk = torch.zeros(
                slam_conf.CHUNK_SIZE, 3, h, w
            )
            self.closing_chunk = torch.zeros(
                2*slam_conf.CHUNK_OVERLAP, 3, h, w
            )

        for i, img in enumerate(imgs):
            self.input_chunk[i] = img

        x = self.input_chunk[:len(kfids)]
        x = self.transform(x)
        with torch.inference_mode():
            descriptors = self.model(x).cpu().numpy()

        self.publish(
            "/vpr/descriptors",
            {
                "chunk_id": self.chunk_cnt,
                "kf_ids": kfids,
                "descriptors": descriptors
            }
        )
        self.logger.info(f"Publishing descriptors of chunk_id: {self.chunk_cnt} with ids: {kfids}")
        self.prev_kfids = kfids
        self.chunk_cnt += 1

    def on_closed_loop(self, msg):
        pass
