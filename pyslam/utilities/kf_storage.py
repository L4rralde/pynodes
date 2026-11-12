from typing import  Dict

import torch


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
