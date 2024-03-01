from threading import Thread

from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import rospy
from tf_conversions import transformations
from scipy.spatial.transform import Rotation
import tf

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from ..base import Base


class RosBase(Base, backend="ros"):

    def __init__(self, backend) -> None:
        super().__init__(backend)
        init_pose = PoseStamped() # Pose = rospy.wait_for_message("/airbot/pose", Pose, timeout=1)
        ###gt###
        # init_pose.pose.position.x = -3.39145622253418
        # init_pose.pose.position.y = -3.3311710357666
        # init_pose.pose.position.z = 0.5880000591278076
        ###hdl###
        init_pose.pose.position.x = -1.2317611908186263
        init_pose.pose.position.y = -0.9177336410440479
        init_pose.pose.position.z = -0.2374505629662929
        
        ###gt###
        # init_pose.pose.orientation.x = 1.1315569281578064e-06
        # init_pose.pose.orientation.y = 3.902353853391105e-07
        # init_pose.pose.orientation.z = 0.9999999403953552
        # init_pose.pose.orientation.w = 3.536624504363317e-08
        
        ###hdl###
        init_pose.pose.orientation.x = 0.006592923324957343
        init_pose.pose.orientation.y = -0.012942241749323641
        init_pose.pose.orientation.z = 0.014944697147015459
        init_pose.pose.orientation.w = 0.9997828203003477
        
        self._target_pose = init_pose
        self._position, self._rotation = self.pose_to_pr(init_pose)

        self.wait_tolerance = np.array([0.2, 0.2], dtype=np.float64)
        self.wait_timeout = 60
        self.wait_period = 0.05

        self.publisher = rospy.Publisher("/airbot/base_pose_cmd", PoseStamped, queue_size=5)
        # self.publisher = rospy.Publisher("/airbot/base_pose_cmd", Pose, queue_size=5)
        self.subscriber = rospy.Subscriber("/airbot/base/current_pose", Pose, self._pose_callback)
        # self.object_subscriber = rospy.Subscriber("/object/poses", Pose, self._obj_pose_callback)
        # self.subscriber = rospy.Subscriber("/airbot/pose", Pose, self._pose_callback)
        self.tf_sub = rospy.Subscriber("/tf", TFMessage, self._tf_callback)

        self.base_pose = None
        self.base_pose_pub = rospy.Publisher("/airbot/base/current_pose", Pose, queue_size=1)
        
        self._raw_init = True
        self._pub_thread = Thread(target=self._publish_task, daemon=True)
        self._pub_thread.start()
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.client.wait_for_server()
        self.goal_pose  = MoveBaseGoal()
        rospy.set_param("/airbot/base/control_finished", False)

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self, *args, **kwargs) -> bool:
        self.inited = False
        self._raw_init = False
        self._pub_thread.join()
        self.subscriber.unregister()
        self.publisher.unregister()
        return True

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray):
        self._position = value

    @property
    def rotation(self) -> float:
        return self._rotation
    
    @rotation.setter
    def rotation(self, value: float):
        self._rotation = value

    @property
    def target_pose(self) -> PoseStamped:
        return self._target_pose
    # def target_pose(self) -> Pose:PoseStamped
    #     return self._target_pose

    @target_pose.setter
    def target_pose(self, value: PoseStamped):
        self._target_pose = value
    # def target_pose(self, value: Pose):
    #     self._target_pose = value

    def move_to(self, position: np.ndarray, rotation: np.ndarray, frame_id: str, avoid_swing: bool) -> tuple:
        rospy.set_param("/airbot/base/avoid_swing", avoid_swing)
        self.target_pose = self.pr_to_pose(position, rotation, frame_id)
        start_time = rospy.get_time()
        print(rospy.get_param("/airbot/base/control_finished", False) )
        # self._publish_task()
        
        while not rospy.get_param("/airbot/base/control_finished", False) and not rospy.is_shutdown():
            # print("Base error: ", self._get_wait_error())
            if (rospy.get_time() - start_time > self.wait_timeout):
                rospy.logerr("Move timeout! The robot may be stuck! The program will continue!")
                break
            rospy.sleep(self.wait_period)
        else:
            rospy.set_param("/airbot/base/control_finished", False)
        # while (self._get_wait_error() > self.wait_tolerance).any() and not rospy.is_shutdown():
        #     if (rospy.get_time() - start_time > self.wait_timeout):
        #         rospy.logerr("Move timeout! The robot may be stuck! The program will continue!")
        #         break
        #     rospy.sleep(self.wait_period)
        return position, rotation

    # def move_to_hdl(self, position: np.ndarray, rotation: np.ndarray):
    #     goal_pose = self.pr_to_pose(position, rotation)
    #     self.goal_pose.target_pose.header.stamp = rospy.Time.now()
    #     self.goal_pose.target_pose.header.frame_id = "map"
    #     self.goal_pose.target_pose.pose.position.x = goal_pose.position.x
    #     self.goal_pose.target_pose.pose.position.y = goal_pose.position.y
    #     self.goal_pose.target_pose.pose.orientation.x = goal_pose.orientation.x
    #     self.goal_pose.target_pose.pose.orientation.y = goal_pose.orientation.y
    #     self.goal_pose.target_pose.pose.orientation.z = goal_pose.orientation.z
    #     self.goal_pose.target_pose.pose.orientation.w = goal_pose.orientation.w
    #     self.client.send_goal(self.goal_pose)
    #     wait = self.client.wait_for_result()
    #     if wait:
    #         if self.client.get_state() != GoalStatus.SUCCEEDED:
    #             print("Base move finished")

    def _publish_task(self):
        rate = rospy.Rate(200)
        while self._raw_init and not rospy.is_shutdown():
            self.publisher.publish(self.target_pose)
            if self.base_pose is not None:
                self.base_pose_pub.publish(self.base_pose)
            rate.sleep()

    def _pose_callback(self, msg: Pose):
        self.position = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float64)
        self.rotation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def _tf_callback(self, cmd_msg):
        for i in cmd_msg.transforms:
            if i.header.frame_id == "3dmap" and i.child_frame_id == "livox_frame":
                # # print(i)
                # p = i.transform
                
                # pos_lidar = np.array([p.translation.x, p.translation.y, p.translation.z])
                # # print(pos_lidar)
                # ori = [p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w]
                # r = Rotation.from_quat(ori)
                # pos = np.array(r.as_matrix()).dot(np.array([-0.31686, 0, -0.26705]).T).T + pos_lidar
                # # pos = np.array(r.as_matrix()).dot(np.array([0, 0, 0]).T).T + pos_lidar
                # # print(pos, ori)
                # # self.position = pos
                # # self.rotation = ori

                from robot_tools.coordinate import CoordinateTools
                car_lidar_pose = (np.array([-0.31686, 0, -0.26705]), np.zeros(3))
                lidar_world_position = np.array([i.transform.translation.x, i.transform.translation.y, i.transform.translation.z])
                lidar_world_orientation = np.array([i.transform.rotation.x, i.transform.rotation.y, i.transform.rotation.z, i.transform.rotation.w])
                lidar_world_pose = (lidar_world_position, lidar_world_orientation)
                car_world_pose = CoordinateTools.to_world_coordinate(car_lidar_pose,lidar_world_pose)
                car_world_pose_list = car_world_pose[0].tolist() + car_world_pose[1].tolist()
                self.base_pose = self._list_to_pose(car_world_pose_list)

    def _get_wait_error(self) -> np.ndarray:
        target_position, target_rotation = self.pose_to_pr(self.target_pose)
        position_error = np.linalg.norm(self.position - target_position)
        current_rotation = transformations.euler_from_quaternion(self.rotation)
        target_rotation = transformations.euler_from_quaternion(target_rotation)
        rotation_error = np.array(target_rotation) - np.array(current_rotation)
        rotation_error[rotation_error > np.pi] -= 2 * np.pi
        rotation_error[rotation_error < -np.pi] += 2 * np.pi
        rotation_error = np.linalg.norm(rotation_error)
        # print(self.position, target_position, position_error, rotation_error)
        return np.array([position_error, rotation_error], dtype=np.float64)

    def _list_to_pose(self, pose_list):
        pose_msg = Pose()
        if len(pose_list) == 7:
            pose_msg.position.x = pose_list[0]
            pose_msg.position.y = pose_list[1]
            pose_msg.position.z = pose_list[2]
            pose_msg.orientation.x = pose_list[3]
            pose_msg.orientation.y = pose_list[4]
            pose_msg.orientation.z = pose_list[5]
            pose_msg.orientation.w = pose_list[6]
        elif len(pose_list) == 6:
            pose_msg.position.x = pose_list[0]
            pose_msg.position.y = pose_list[1]
            pose_msg.position.z = pose_list[2]
            q = tf.transformations.quaternion_from_euler(
                pose_list[3], pose_list[4], pose_list[5]
            )
            pose_msg.orientation.x = q[0]
            pose_msg.orientation.y = q[1]
            pose_msg.orientation.z = q[2]
            pose_msg.orientation.w = q[3]
        else:
            raise Exception(
                "Expected either 6 or 7 elements in list: (x,y,z,r,p,y) or (x,y,z,qx,qy,qz,qw)"
            )
        return pose_msg


    @staticmethod
    def pr_to_pose(position, rotation, id):
        pose = PoseStamped()
        pose.header.frame_id = id
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        if len(rotation) == 3:
            rotation = transformations.quaternion_from_euler(rotation[0], rotation[1], rotation[2])
        pose.pose.orientation.x = rotation[0]
        pose.pose.orientation.y = rotation[1]
        pose.pose.orientation.z = rotation[2]
        pose.pose.orientation.w = rotation[3]
        return pose

    @staticmethod
    def pose_to_pr(pose):
        position = np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ], dtype=np.float64)
        rotation = np.array([
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ],
                            dtype=np.float64)
        return position, rotation
