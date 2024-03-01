from copy import deepcopy
import threading

from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import rospy

from ..base import Arm


class RosArm(Arm, backend="ros"):

    def _posestamed_to_np(self, msg: PoseStamped) -> tuple:
        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        orientation = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        return position, orientation

    def _np_to_posestamed(self, position: np.ndarray, orientation: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        # Construct PoseStamped message with current time
        msg.header.stamp = rospy.get_rostime()
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        return msg

    def _np_to_pose(self, position:np.ndarray, orientation: np.ndarray) -> Pose:
        msg = Pose()
        msg.position.x = position[0]
        msg.position.y = position[1]
        msg.position.z = position[2]
        msg.orientation.x = orientation[0]
        msg.orientation.y = orientation[1]
        msg.orientation.z = orientation[2]
        msg.orientation.w = orientation[3]
        return msg

 
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self._end_pose: tuple = None
        self._end_pose_lock = threading.Lock()
        self._target_pose: tuple = None
        self._target_pose_lock = threading.Lock()

        # self.end_pose = np.array([0.12, 0, 0.15]), np.array([0., 0., 0., 1.])
        # Initialize the end pose and target pose variable with
        # the first message (current pose obtained from topic)
        self._on_pose_msg(rospy.wait_for_message("/airbot_play/current_pose", PoseStamped))
        self._subscribe_thread = rospy.Subscriber("/airbot_play/current_pose", PoseStamped, self._on_pose_msg)
        target_pose = deepcopy(self.end_pose)
        self.target_pose = target_pose

        self._publisher = rospy.Publisher("/airbot_play/pose_cmd", PoseStamped, queue_size=5)
        self._publish_thread = threading.Thread(target=self._pub_pose, daemon=True)
        self._publish_thread.start()

        # Parameters for checking if target has been reached
        self._wait_tolerance = np.array([0.02, 0.1])
        self._wait_timeout = 30
        self._wait_period = 0.01

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self, *args, **kwargs) -> bool:
        self.inited = False
        return True

    @property
    def end_pose(self) -> tuple:
        self._end_pose_lock.acquire()
        value = deepcopy(self._end_pose)
        self._end_pose_lock.release()
        return value

    @end_pose.setter
    def end_pose(self, value: tuple):
        self._end_pose_lock.acquire()
        self._end_pose = value
        self._end_pose_lock.release()

    @property
    def target_pose(self) -> tuple:
        self._target_pose_lock.acquire()
        value = deepcopy(self._target_pose)
        self._target_pose_lock.release()
        return value

    @target_pose.setter
    def target_pose(self, value: tuple):
        self._target_pose_lock.acquire()
        self._target_pose = value
        self._target_pose_lock.release()

    def move_end_to_pose(self, position: np.ndarray, orientation: np.ndarray) -> tuple:
        self.target_pose = (position, orientation)
        start_time = rospy.get_time()
        
        self._pub_pose()

        while (self._get_wait_error() > self._wait_tolerance).any():
            if (rospy.get_time() - start_time > self._wait_timeout):
                rospy.logerr("Move timeout! The robot may be stuck! The program will continue!")
                break
            rospy.sleep(self._wait_period)
        return position, orientation

    def _pub_pose(self):
        rate = rospy.Rate(200)
        if not rospy.is_shutdown():
            self._publisher.publish(self._np_to_posestamed(*self.target_pose))
            rate.sleep()

    def _on_pose_msg(self, msg: PoseStamped):
        self.end_pose = self._posestamed_to_np(msg)

    def _get_wait_error(self) -> np.ndarray:
        target_position, target_orientation = self.target_pose
        position_error = np.linalg.norm(self.end_pose[0] - target_position)
        orientation_error = np.linalg.norm(self.end_pose[1] - target_orientation)
        return np.array([position_error, orientation_error], dtype=np.float64)
