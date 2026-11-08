from typing import Dict, List

import torch
import torchvision.transforms.v2 as T
import numpy as np

from pynodes import Node, Message
import pyslam.conf as slam_conf


#FIXME: This node is not tested and some bugs are known.
class FoundationModelNode(Node):
    name = "3dgfm"
    class KeyFrameStorage:
        def __init__(self):
            self.kf_dict: Dict[torch.Tensor] = {}

        def add(self, kf_id: int, img: torch.Tensor) -> None:
            if kf_id in self.kf_dict.keys():
                raise ValueError("Kf id already in storage")
            self.kf_dict[kf_id] = img

        def get(self, kf_id: int) -> torch.Tensor:
            if not kf_id in self.kf_dict.keys():
                raise RuntimeError(f"Unknown kf_id: {kf_id}")

            return self.kf_dict[kf_id]

    def __init__(self, node_id, inbox, outbound, stop_event):
        super().__init__(node_id, inbox, outbound, stop_event)
        self.kf_storage = FoundationModelNode.KeyFrameStorage()
        self.input_chunk: torch.Tensor|None = None
        self.closing_chunk: torch.Tensor|None = None
        self.model: torch.nn.Module = None #self.load_model().eval().to('cuda')
        self.chunk_cnt: int=0
        self.prev_kfids: List[int] = []

    def setup(self):
        self.subscribe("/video/chunks", self.on_chunk)
        self.subscribe("/closed_loop/closing_chunk", self.on_closed_loop)
        self.advertise("/3dgfm/predictions")
        self.advertise("/vpr/descriptors")
        #self.model = self.load_model().eval().to('cuda')

    def to_tensor_img(self, np_img: np.ndarray) -> torch.Tensor:
        return T.ToTensor()(np_img)

    def model_prediction(self, images: torch.Tensor, extract_features: bool=True) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            predictions = self.model.forward(images)
        
        extrinsics, intrinsics = self.model.backbone.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        predictions['extrinsic'] = extrinsics
        predictions['intrinsic'] = intrinsics

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        if not extract_features:
            predictions.pop('descriptor', None) #Skip computation if not required

        return predictions

    def load_model(self) -> torch.nn.Module:
        model = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            'vggto_s_pre',
            slam_conf.VPR_REPO_PATH,
            vggto_omega_ckpt=slam_conf.VGGT_OMEGA_CKPT
        )
        return model

    def on_chunk(self, msg: Message) -> None:
        kfids = msg.payload['kf_ids']
        imgs = msg.payload['imgs']

        imgs = [self.to_tensor_img(img) for img in imgs]

        for kfid, img in zip(kfids, imgs):
            self.kf_storage.add(kfid, img)

        h, w = imgs[0].shape[:2]
        if self.input_chunk is None:
            self.input_chunk = torch.zeros(
                slam_conf.CHUNK_SIZE, 3, h, w
            )
            self.closing_chunk = torch.zeros(
                2*slam_conf.CHUNK_OVERLAP, 3, h, w
            )

        for i, img in enumerate(imgs):
            self.input_chunk[i] = img

        predictions = self.model_prediction(self.input_chunk.to('cuda'))
        descriptors = predictions.pop("descriptor")
        self.publish(
            "/vpr/descriptors",
            {
                "chunk_id": self.chunk_cnt,
                "kf_ids": kfids,
                "descriptors": descriptors
            }
        )

        self.publish(
            "/3dgfm/predictions",
            {
                "type": "measurement",
                "chunk_id": self.chunk_cnt,
                "kf_ids": kfids,
                "predictions": predictions,
            }
        )

        self.chunk_cnt += 1

    def on_closed_loop(self, msg: Message) -> None:
        kfids = msg.payload['kf_ids']
        for i, kf_id in enumerate(kfids):
            self.closing_chunk[i] = self.kf_storage.get(kf_id)

        predictions = self.model_prediction(self.closing_chunk.to('cuda'))
        self.publish(
            "/3dgfm/predictions",
            {
                "type": "loop_closure",
                'chunk_id': None,
                'kf_ids': kfids,
                'predictions': predictions,
                'edge_info': {
                    'sim': msg.payload['sim'],
                    'parent': msg.payload['parent'],
                    'child': msg.payload['child']
                }
            }
        )


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

