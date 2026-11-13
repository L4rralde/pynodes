from typing import Dict, Any

import numpy as np

from .sharray import AllocatedSharedMemoryNdarray


class BufferData:
        def __init__(self, buffers):
            self.buffers = buffers

        def close(self) -> None:
            for b in self.buffers:
                b.close()


class AllocatedShmPcdData:
    def __init__(self, num_imgs: int):
        self.points = AllocatedSharedMemoryNdarray(
            "pointcloud_points",
            max_num_bytes=num_imgs*512*512*3*8
        )
        self.colors = AllocatedSharedMemoryNdarray(
            "pointcloud_colors",
            max_num_bytes=num_imgs*512*512*3*8
        )

    def fill(self, points: np.ndarray, colors: np.ndarray) -> None:
        self.points.fill(points)
        self.colors.fill(colors)

    def get_metadata(self) -> None:
        return {
            "points": self.points.get_metadata(),
            "colors": self.colors.get_metadata()
        }

    @staticmethod
    def attach(metadata: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        attached_data = {
            k: AllocatedSharedMemoryNdarray.attach(v)
            for k, v in metadata.items()
        }

        buffers = [v[0] for v in attached_data.values()]

        data_dict = {
            k: v[1].copy()
            for k,v in attached_data.items()
        }

        return BufferData(buffers), data_dict
