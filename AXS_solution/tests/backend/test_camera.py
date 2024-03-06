import numpy as np
import pytest

from airbot.backend import Camera


@pytest.mark.parametrize("backend", ('mock', 'ros'))
class TestCamera:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, backend):
        self.camera = Camera(backend=backend)
        assert self.camera.init()
        yield
        assert self.camera.deinit()

    def test_init(self):
        assert self.camera.inited

    def test_get_rgb(self):
        rgb = self.camera.get_rgb()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (self.camera.HEIGHT, self.camera.WIDTH, 3)
        assert rgb.dtype == np.uint8

    def test_get_depth(self):
        depth = self.camera.get_depth()
        assert isinstance(depth, np.ndarray)
        assert depth.shape == (self.camera.HEIGHT, self.camera.WIDTH)
        assert depth.dtype == np.float32

    def test_get_intrinsic(self):
        intrinsic = self.camera.INTRINSIC
        assert isinstance(intrinsic, np.ndarray)
        assert intrinsic.shape == (3, 3)
        assert intrinsic.dtype == np.float64
