import numpy as np

from ..base import Base


class MockBase(Base, backend="mock"):
    INIT_POSITION = np.array([0, 0, 0], dtype=np.float64)
    INIT_ROTATION = np.array([0, 0, 0], dtype=np.float64)
    INIT_WAIT_TOLERANCE = np.array([0.1, 0.1], dtype=np.float64)

    def __init__(self, backend) -> None:
        super().__init__(backend)
        self._position = self.INIT_POSITION
        self._rotation = self.INIT_ROTATION
        self._wait_tolerance = self.INIT_WAIT_TOLERANCE

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self, *args, **kwargs) -> bool:
        self.inited = False
        return True

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation

    @property
    def wait_tolerance(self) -> np.ndarray:
        return self._wait_tolerance

    def move_to(self, position: np.ndarray, rotation: np.ndarray) -> tuple:
        self._position = position
        self._rotation = rotation
        return self.position, self.rotation
