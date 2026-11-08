import os
from typing import List
from glob import glob
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event as EventClass
import time

from PIL import Image
import numpy as np
import cv2


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    cv_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(cv_frame_rgb)
    return pil_frame


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(
        np.asarray(img.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )


def pil_to_ndarray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def sort_files_numerically(file_list):
    return sorted(
        file_list,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )


class VideoPublisher(ABC):
    @abstractmethod
    def __init__(self, path: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_fps(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError()


class FolderPublisher(VideoPublisher):
    def __init__(self, path: str, fps: float=-1):
        self.fpaths = self.load_files(path)
        if not self.fpaths:
            raise ValueError(f"Found no image in folder: {path}")
        self.fps = fps
        self.idx_to_publish: int = 0

    def load_files(self, dir: str) -> List[str]:
        fpaths = glob(os.path.join(dir, '*.jpg')) + glob(os.path.join(dir, '*.png'))
        fpaths = sort_files_numerically(fpaths)
        return fpaths

    def get_fps(self) -> float:
        return self.fps

    def read(self) -> np.ndarray:
        if self.idx_to_publish == len(self.fpaths):
            return None
        fpath = self.fpaths[self.idx_to_publish]
        img = pil_to_ndarray(Image.open(fpath))
        self.idx_to_publish += 1
        return img

    def release(self) -> None:
        self.idx_to_publish = 0

class TimedVideoPublisher(FolderPublisher):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

        self.first_time_published: float | None = None

        if self.fps <= 0:
            self.fps = 30

    def read_by_idx(self, idx: int) -> np.ndarray:
        fpath = self.fpaths[idx]

        with Image.open(fpath) as image:
            return pil_to_ndarray(image)

    def read(self) -> np.ndarray | None:
        if self.idx_to_publish >= len(self.fpaths):
            return None

        if self.first_time_published is None:
            self.first_time_published = time.monotonic()

        while True:
            current_time = time.monotonic()
            elapsed = current_time - self.first_time_published

            next_idx = int(elapsed * self.fps)

            if next_idx < self.idx_to_publish:
                next_frame_time = (
                    self.first_time_published
                    + self.idx_to_publish / self.fps
                )
                time.sleep(max(0, next_frame_time - current_time))
                continue

            if next_idx >= len(self.fpaths):
                return None

            if next_idx > self.idx_to_publish:
                print(
                    f"Warning, skipping frames: "
                    f"{list(range(self.idx_to_publish, next_idx))}"
                )

            self.idx_to_publish = next_idx + 1

            return self.read_by_idx(next_idx)


class VideoCapturePublisher(VideoPublisher):
    def __init__(self, path: str):
        self.video = cv2.VideoCapture(path)

    def get_fps(self) -> int:
        fps = self.video.get(cv2.CAP_PROP_FPS)
        if not fps:
            return -1
        return fps

    def read(self) -> np.ndarray:
        ret, frame = self.video.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self.video.release()


def get_publisher(path: str, **kwargs) -> VideoPublisher:
    if os.path.isdir(path):
        return FolderPublisher(path, **kwargs)
    if path.endswith('.mp4'):
        return VideoCapturePublisher(path, **kwargs)
    raise ValueError(f"Unrecognize type of path: {path}")
