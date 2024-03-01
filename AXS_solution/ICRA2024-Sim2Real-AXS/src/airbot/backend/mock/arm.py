import numpy as np

from ..base import Arm


class MockArm(Arm, backend='mock'):

    def __init__(self, backend) -> None:
        super().__init__(backend)
        self._end_pose: tuple = self.INIT_END_POSE
        self._wait_tolerance = self.INIT_WAIT_TOLERANCE

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self, *args, **kwargs) -> bool:
        self.inited = False
        return True

    @property
    def INIT_END_POSE(self) -> np.ndarray:
        return np.array([0, 0, 0], dtype=np.float64), np.array([0, 0, 0, 1], dtype=np.float64)

    @property
    def INIT_WAIT_TOLERANCE(self) -> np.ndarray:
        return np.array([0.1, 0.1], dtype=np.float64)

    @property
    def end_pose(self) -> np.ndarray:
        return self._end_pose

    @property
    def wait_tolerance(self) -> np.ndarray:
        return self._wait_tolerance

    def move_end_to_pose(self, position: np.ndarray, orientation: np.ndarray) -> tuple:
        self._end_pose = position, orientation
        return self._end_pose
