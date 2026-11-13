
from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

import numpy as np
import torch


class AllocatedImgChunk(ABC):
    def __init__(self, max_chunk_size: int = 32):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

        self.data = None
        self.h = -1
        self.w = -1

        self.max_chunk_size = max_chunk_size
        self.num_valid_entries = 0
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    def initialize(self):
        pass

    def set_img_shape(self, img_shape: Tuple[int, int]) -> None:
        if self._initialized:
            raise RuntimeError("Chunk is already initialized")

        if len(img_shape) < 2:
            raise ValueError("Image shape must contain H and W")

        self.h, self.w = img_shape[:2]

        if self.h <= 0 or self.w <= 0:
            raise ValueError("Image dimensions must be positive")

        self.data = self.initialize()
        self._initialized = True

    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Chunk has not been initialized")

    def flush(self) -> None:
        self._check_initialized()
        self.num_valid_entries = 0

    def fill(self, input_list: Sequence[Any]) -> None:
        self._check_initialized()

        if len(input_list) > self.max_chunk_size:
            raise ValueError(
                f"Input size {len(input_list)} exceeds "
                f"capacity {self.max_chunk_size}"
            )

        for i, datum in enumerate(input_list):
            self[i] = datum

        self.num_valid_entries = len(input_list)

    def get(self):
        self._check_initialized()
        return self.data[:self.num_valid_entries]

    def __len__(self) -> int:
        return self.num_valid_entries

    def __setitem__(self, key, value):
        self._check_initialized()
        self.data[key] = value

    def __getitem__(self, key):
        self._check_initialized()
        return self.data[key]

    def append(self, value) -> None:
        self._check_initialized()

        if self.num_valid_entries >= self.max_chunk_size:
            raise RuntimeError("Allocated data is full")

        self[self.num_valid_entries] = value
        self.num_valid_entries += 1


class AllocatedTorchImgChunk(AllocatedImgChunk):
    def __init__(
        self,
        max_chunk_size: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(max_chunk_size)
        self.device = torch.device(device)
        self.dtype = dtype

    def initialize(self) -> torch.Tensor:
        return torch.zeros(
            self.max_chunk_size,
            3,
            self.h,
            self.w,
            device=self.device,
            dtype=self.dtype,
        )

    def __setitem__(self, key, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")

        value = value.to(
            device=self.device,
            dtype=self.dtype,
        )

        super().__setitem__(key, value)


class AllocatedNumpyImgChunk(AllocatedImgChunk):
    def initialize(self) -> np.ndarray:
        return np.zeros(
            (self.max_chunk_size, self.h, self.w, 3),
            dtype=np.uint8,
        )
