from typing import List, Tuple

import torch


class AllocatedTorchImgChunk:
    def __init__(self, max_chunk_size: int=32, device: str="cuda"):
        self.tensor: torch.Tensor|None = None
        self.h: int = -1
        self.w: int = -1
        self.max_chunk_size: int = max_chunk_size
        self.num_valid_entries: int = -1
        self.device = device
        self._initialized: bool=False

    def is_initialized(self) -> bool:
        return self._initialized


    def set_img_shape(self, img_shape: Tuple[int]) -> None:
        self.h, self.w = img_shape[:2]
        self.tensor = torch.zeros(
            self.max_chunk_size, 3, self.h, self.w
        ).to(self.device)
        self._initialized = True


    def fill(self, input_tensor_list: List[torch.Tensor]) -> None:
        for i, tensor in enumerate(input_tensor_list):
            self.tensor[i] = tensor.to(self.device)
        self.num_valid_entries = len(input_tensor_list)

    def get(self) -> torch.Tensor:
        return self.tensor[:self.num_valid_entries]

