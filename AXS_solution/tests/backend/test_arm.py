import numpy as np
import pytest

from airbot.backend import Arm


@pytest.mark.parametrize("backend", ('mock', 'ros'))
class TestArm():
    POSITION_THRESHOLD = np.array([0.05, 0.05, 0.05])
    POSE_THRESHOLD = np.array([0.05, 0.05, 0.05, 0.05])

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, backend):
        self.arm = Arm(backend=backend)
        assert self.arm.init()
        yield
        assert self.arm.deinit()

    def test_init(self):
        assert self.arm.inited
        assert (self.arm.end_pose[0] == self.arm.INIT_END_POSE[0]).all()
        assert (self.arm.end_pose[1] == self.arm.INIT_END_POSE[1]).all()

    def test_get_pose(self):
        pose = self.arm.end_pose
        assert isinstance(pose, tuple)
        assert pose[0].shape == (3,)
        assert pose[1].shape == (4,)

    @pytest.mark.parametrize("pose", (
        (np.array([0.62, -0.17, 0.68], dtype=np.float64), np.array([0, 0, 0, 1], dtype=np.float64)),
        (np.array([0.682, -0.33, 0.72], dtype=np.float64), np.array([0.345, 0.198, -0.12, 0.909], dtype=np.float64)),
        (np.array([0.62, -0.17, 0.568], dtype=np.float64), np.array([0, 0, 0, 1], dtype=np.float64)),
    ))
    def test_move_to_pose(self, pose):
        self.arm.move_end_to_pose(pose[0], pose[1])
        assert (np.absolute(self.arm.end_pose[0] - pose[0]) - self.POSITION_THRESHOLD <= 0).all()
        assert (np.absolute(self.arm.end_pose[1] - pose[1]) - self.POSE_THRESHOLD <= 0).all()
