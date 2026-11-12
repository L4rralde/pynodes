import multiprocessing.shared_memory as shared_memory
from typing import Dict, Any
from dataclasses import dataclass

import numpy as np

from vggt_salad_slam.dtypes import Prediction


DEF_MAX_SIZE = 4 * 1024 * 1024


class AllocatedSharedMemoryNdarray:
    def __init__(self, name: str, max_num_bytes: int=DEF_MAX_SIZE):
        self.shm = shared_memory.SharedMemory(
            name=name,
            create=True,
            size=max_num_bytes
        )
        self.name = name
        self.max_num_bytes = max_num_bytes
        self.shared_array: np.ndarray|None = None
        self.metadata: Dict[str, Any] = {}

    def fill(self, array: np.ndarray) -> None:
        if array.nbytes >= self.max_num_bytes:
            raise RuntimeError(f"{self.name} shm array exceeded num of bytes")
        self.metadata = {
            "name": self.name,
            "shape": array.shape,
            "dtype": array.dtype.str
        }

        self.shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=self.shared_array
        )

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    @staticmethod
    def attach(metadata) -> np.ndarray:
        shm = shared_memory.SharedMemory(
            name = metadata["name"]
        )

        array = np.ndarray(
            shape=metadata["shape"],
            dtype=np.dtype(metadata["dtype"]),
            buffer=shm.buf
        )

        return array


class AllocatedShmPrediction:
    def __init__(self) -> None:
        self.depth = AllocatedSharedMemoryNdarray("prediction_depth")
        self.depth_conf = AllocatedSharedMemoryNdarray("prediction_depth_conf")
        self.extrinsic = AllocatedSharedMemoryNdarray("prediction_extrinsic", max_num_bytes=8192)
        self.intrinsic = AllocatedSharedMemoryNdarray("prediction_intrinsic", max_num_bytes=4096)
        self.images = AllocatedSharedMemoryNdarray("prediction_images")
        self.ids = AllocatedSharedMemoryNdarray("predictions_ids", max_num_bytes=512)
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
            "extrinsic": self.depth_conf.get_metadata(),
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
        pred_dict = {
            k: AllocatedSharedMemoryNdarray.attach(v)
            for k,v in metadata.items()
        }
        return Prediction.from_dict(pred_dict)
