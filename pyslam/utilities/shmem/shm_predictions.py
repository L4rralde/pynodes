from typing import Dict, Any

import numpy as np

from vggt_salad_slam.dtypes import Prediction
from .sharray import AllocatedSharedMemoryNdarray


ONE_IMG_BYTES = 512*512*3*4


class AllocatedShmPrediction:
    def __init__(self, num_imgs: int) -> None:
        self.depth = AllocatedSharedMemoryNdarray(
            "prediction_depth",
            max_num_bytes=num_imgs*ONE_IMG_BYTES
        )
        self.depth_conf = AllocatedSharedMemoryNdarray(
            "prediction_depth_conf",
            max_num_bytes=num_imgs*ONE_IMG_BYTES
        )
        self.extrinsic = AllocatedSharedMemoryNdarray(
            "prediction_extrinsic",
            max_num_bytes=256*num_imgs
        )
        self.intrinsic = AllocatedSharedMemoryNdarray(
            "prediction_intrinsic",
            max_num_bytes=256*num_imgs
        )
        self.images = AllocatedSharedMemoryNdarray(
            "prediction_images",
            max_num_bytes=num_imgs*ONE_IMG_BYTES
        )
        self.ids = AllocatedSharedMemoryNdarray(
            "predictions_ids",
            max_num_bytes=num_imgs*32
        )
        self.ids_not_null: bool = False


    def fill(self, preds: Prediction) -> None:
        self.depth.fill(preds.depth)
        self.depth_conf.fill(preds.depth_conf)
        self.extrinsic.fill(preds.extrinsic)
        self.intrinsic.fill(preds.intrinsic)
        self.images.fill(preds.images)
        self.ids_not_null = preds.ids is not None
        if self.ids_not_null:
            self.ids.fill(np.asarray(preds.ids))

    def get_metadata(self) -> None:
        mandatory_metadata = {
            "depth": self.depth.get_metadata(),
            "depth_conf": self.depth_conf.get_metadata(),
            "extrinsic": self.extrinsic.get_metadata(),
            "intrinsic": self.intrinsic.get_metadata(),
            "images": self.images.get_metadata(),
        }
        opt_metadata = {}
        if self.ids_not_null:
            opt_metadata = {
                "ids": self.ids.get_metadata()
            }

        metadata = {
            **mandatory_metadata,
            **opt_metadata
        }

        return metadata

    @staticmethod
    def attach(metadata: Dict[str, Dict[str, Any]]) -> Prediction:
        attached_data = {
            k: AllocatedSharedMemoryNdarray.attach(v)
            for k, v in metadata.items()
        }
        buffers = [v[0] for v in attached_data.values()]
        pred_dict ={
            k: v[1].copy()
            for k,v in attached_data.items()
        }
        for b in buffers:
            b.close()

        return Prediction.from_dict(pred_dict)
