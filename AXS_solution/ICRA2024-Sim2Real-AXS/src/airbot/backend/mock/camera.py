import cv2
from importlib_resources import files
import numpy as np

from ..base import Camera


class MockCamera(Camera, backend='mock'):

    _WIDTH = 480
    _HEIGHT = 270
    _INTRINSIC = np.array([
        [609.765, 0.0, 322.594],
        [0.0, 608.391, 243.647],
        [0.0, 0.0, 1.0],
    ])

    @property
    def WIDTH(self) -> int:
        return self._WIDTH

    @property
    def HEIGHT(self) -> int:
        return self._HEIGHT

    @property
    def INTRINSIC(self) -> int:
        return self._INTRINSIC

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self) -> bool:
        self.inited = False
        return True

    def get_rgb(self) -> np.ndarray:
        img_path = files('airbot.data').joinpath('mock_rgb.png')
        img = cv2.cvtColor(cv2.imread(img_path.as_posix()), cv2.COLOR_BGR2RGB)
        return img

    def get_depth(self) -> np.ndarray:
        img_path = files('airbot.data').joinpath('mock_depth.png')
        img = cv2.cvtColor(cv2.imread(img_path.as_posix()), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return img
