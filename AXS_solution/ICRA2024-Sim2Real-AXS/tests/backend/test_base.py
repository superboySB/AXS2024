import numpy as np
import pytest

from airbot.backend import Base


@pytest.mark.parametrize("backend", ('mock', 'ros'))
class TestBase():
    POSITION_THRESHOLD = np.array([0.1, 0.1, 0.1])
    ROTATION_THRESHOLD = np.array([0.1, 0.1, 0.1])

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, backend):
        self.base = Base(backend=backend)
        assert self.base.init()
        yield
        assert self.base.deinit()

    def test_init(self):
        assert self.base.inited
        assert (self.base.position == self.base.INIT_POSITION).all()
        assert (self.base.rotation == self.base.INIT_ROTATION).all()

    def test_get_position(self):
        position = self.base.position
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert position.dtype == np.float64

    def test_get_rotation(self):
        rotation = self.base.rotation
        assert isinstance(rotation, np.ndarray)
        assert rotation.shape == (3,)
        assert rotation.dtype == np.float64

    @pytest.mark.parametrize("position", (
        np.array([-5.13, -3.49, 0.513], dtype=np.float64),
        np.array([-4, -3.5, 0], dtype=np.float64),
        np.array([-3.5, -3.5, 3], dtype=np.float64),
    ))
    def test_move_to_position(self, position):
        self.base.move_to(position, self.base.rotation)
        assert (np.absolute(self.base.position - position) - self.POSITION_THRESHOLD <= 0).all()

    @pytest.mark.parametrize("rotation", (
        np.array([0, 0, -0.5], dtype=np.float64),
        np.array([5, 2, 1], dtype=np.float64),
        np.array([1, 2, -3], dtype=np.float64),
    ))
    def test_move_to_rotation(self, rotation):
        self.base.move_to(self.base.position, rotation)
        assert (self.base.rotation == rotation).all()
        assert (np.absolute(self.base.rotation - rotation) - self.ROTATION_THRESHOLD <= 0).all()
