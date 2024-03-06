import numpy as np
import pytest

from airbot.backend import Gripper


@pytest.mark.parametrize("backend", ('mock', 'ros'))
class TestGripper():

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, backend):
        self.gripper = Gripper(backend=backend)
        assert self.gripper.init()
        yield
        assert self.gripper.deinit()

    def test_init(self):
        assert self.gripper.inited
        assert self.gripper.state == Gripper.Status.CLOSE

    def test_open(self):
        assert self.gripper.open() == Gripper.Status.OPEN
        assert self.gripper.state == Gripper.Status.OPEN

    def test_close(self):
        assert self.gripper.close() == Gripper.Status.CLOSE
        assert self.gripper.state == Gripper.Status.CLOSE

    def test_open_close(self):
        for i in range(10):
            self.gripper.open()
            self.gripper.close()
        assert self.gripper.state == Gripper.Status.CLOSE
