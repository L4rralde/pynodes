from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import torch
import numpy as np


class AllocatedImgChunk(ABC):
    def __init__(self, max_chunk_size: int=32):
        self.data = None
        self.h: int = -1
        self.w: int = -1
        self.max_chunk_size: int = max_chunk_size
        self.num_valid_entries: int = -1
        self._initialized: bool=False

    def is_initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def initialize(self) -> object:
        raise NotImplementedError

    def flush(self) -> None:
        self.num_valid_entries = 0

    def set_img_shape(self, img_shape: Tuple[int]) -> None:
        self.h, self.w = img_shape[:2]
        self.data = self.initialize()
        self._initialized = True

    def fill(self, input_list: List[Any]) -> None:
        for i, datum in enumerate(input_list):
            self.data[i] = datum
        self.num_valid_entries = len(input_list)

    def get(self) -> object:
        return self.data[:self.num_valid_entries]

    def __len__(self) -> int:
        return self.num_valid_entries

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def append(self, value) -> None:
        if self.num_valid_entries == self.max_chunk_size:
            raise RuntimeError("Allocated data is full")
        self.data[self.num_valid_entries] = value
        self.num_valid_entries += 1


class AllocatedTorchImgChunk(AllocatedImgChunk):
    def __init__(self, max_chunk_size = 32, device: str="cuda"):
        super().__init__(max_chunk_size)
        self.device = device

    def initialize(self) -> torch.Tensor:
        return torch.zeros(
            self.max_chunk_size, 3, self.h, self.w
        ).to(self.device)
    
    def fill(self, input_list: List[torch.Tensor]):
        for i, datum in enumerate(input_list):
            self.data[i] = datum.to(self.device)
        self.num_valid_entries = len(input_list)

    def __setitem__(self, key, value):
        return super().__setitem__(key, value.to(self.device))

class AllocatedNumpyImgChunk(AllocatedImgChunk):
    def initialize(self) -> np.ndarray:
        return np.zeros(
            (self.max_chunk_size, self.h, self.w, 3),
            dtype=np.uint8
        )
