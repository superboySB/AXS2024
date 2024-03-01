from threading import Thread

import rospy
from std_msgs.msg import Bool,String

from ..base import Gripper


class RosGripper(Gripper, backend="ros"):

    def __init__(self, backend) -> None:
        super().__init__(backend)
        self._state_callback(rospy.wait_for_message("/airbot_play/gripper/current_state", String))
        self._state_cmd = self.state
        self._cmd_puber = rospy.Publisher("/airbot_play/gripper/state_cmd", Bool, queue_size=5)
        self._state_suber = rospy.Subscriber("/airbot_play/gripper/current_state", String, self._state_callback, queue_size=5)
        self._state_msg = Bool()
        self._state_msg.data = self.state
        self._status_to_bool = {Gripper.Status.CLOSE: True, Gripper.Status.OPEN: False}
        self._raw_init = True
        self._pub_thread = Thread(target=self._cmd_pub_thread, daemon=True)
        self._pub_thread.start()

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self) -> bool:
        self.inited = False
        self._raw_init = False
        self._pub_thread.join()
        self._state_suber.unregister()
        self._cmd_puber.unregister()
        return True

    def open(self) -> Gripper.Status:
        self._state_cmd = Gripper.Status.OPEN
        return self._state_cmd

    def close(self) -> Gripper.Status:
        self._state_cmd = Gripper.Status.CLOSE
        return self._state_cmd

    @property
    def state(self) -> Gripper.Status:
        return self._state

    def _cmd_pub_thread(self):
        rate = rospy.Rate(200)
        while self._raw_init and not rospy.is_shutdown():
            if self._state_cmd != Gripper.Status.MOVING:
                self._state_msg.data = self._status_to_bool[self._state_cmd]
                self._cmd_puber.publish(self._state_msg)
            rate.sleep()

    def _state_callback(self,msg:String):
        if msg.data == "open":
            self._state = Gripper.Status.OPEN
        elif msg.data == "close":
            self._state = Gripper.Status.CLOSE
        elif msg.data == "moving":
            self._state = Gripper.Status.MOVING
