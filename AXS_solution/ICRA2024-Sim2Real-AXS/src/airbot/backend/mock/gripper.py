from ..base import Gripper


class MockGripper(Gripper, backend='mock'):

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        self._state = Gripper.Status.CLOSE
        return True

    def deinit(self) -> bool:
        self.inited = False
        return True

    def open(self) -> Gripper.Status:
        self._state = Gripper.Status.OPEN
        return self._state

    def close(self) -> Gripper.Status:
        self._state = Gripper.Status.CLOSE
        return self._state

    @property
    def state(self) -> Gripper.Status:
        return self._state
