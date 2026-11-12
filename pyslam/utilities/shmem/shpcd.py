from typing import Dict, Any

import numpy as np

from .sharray import AllocatedSharedMemoryNdarray


class AllocatedShmPcdData:
    def __init__(self):
        self.points = AllocatedSharedMemoryNdarray("pointcloud_points")
        self.colors = AllocatedSharedMemoryNdarray("pointcloud_colors")

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

        for b in buffers:
            b.close()

        return data_dict
