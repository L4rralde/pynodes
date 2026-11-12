import multiprocessing.shared_memory as shared_memory
from typing import Dict, Any

import numpy as np

from vggt_salad_slam.dtypes import Prediction


DEF_MAX_SIZE = 32 * 1024 * 1024


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
        if array.nbytes > self.max_num_bytes:
            raise RuntimeError(
                f"{self.name} shm array exceeded num of bytes. {array.nbytes} > {self.max_num_bytes}"
            )
        self.metadata = {
            "name": self.name,
            "shape": array.shape,
            "dtype": array.dtype.str
        }

        self.shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=self.shm.buf
        )
        self.shared_array[:] = array

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def release(self) -> None:
        if self.shared_array is not None:
            del self.shared_array
            self.shared_array = None

        self.shm.close()
        #self.shm.unlink()

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

        return shm, array



