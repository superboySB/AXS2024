
from .srv._airbot_play_ik import airbot_play_ik, airbot_play_ikRequest, airbot_play_ikResponse
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from geometry_msgs.msg import PoseStamped
import rospy


class AIRbotPlayIkServer:

    def __init__(self) -> None:
        self.ik_server = rospy.Service('/airbot_play/ik_service', airbot_play_ik, self._handle_ik)
        self._moveit_ik_service = rospy.ServiceProxy('/airbot_play/compute_ik', GetPositionIK)
        self._target_joint = None
        self.funcs = {}

    def set_init_joint(self, init_joint:tuple):
        self._target_joint = init_joint

    def get_target_joint(self):
        return self._target_joint

    def _handle_ik(self, req:airbot_play_ikRequest):
        target_joint = self._compute_inverse_kinematics(req.target_pose)
        if target_joint is not None:
            self._target_joint = target_joint
            result = 1
        else:
            result = 0
        response = airbot_play_ikResponse()
        response.result = result
        for func in self.funcs.values():
            func(target_joint)
        return response

    def _compute_inverse_kinematics(self, target_pose:PoseStamped):
        """ 逆运动学：将pose(x,y,z,x,y,z,w)转换为joint角度目标 """
        ik_request = GetPositionIKRequest()
        ik_request.ik_request.group_name = "airbot_play_arm"
        ik_request.ik_request.ik_link_name = "custom_end_link"
        ik_request.ik_request.pose_stamped = target_pose
        ik_request.ik_request.pose_stamped.header.frame_id = "arm_base"
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.timeout = rospy.Duration(1)
        ik_response:GetPositionIKResponse = self._moveit_ik_service.call(ik_request)
        if ik_response.error_code.val == ik_response.error_code.SUCCESS:
            target_joint_names = ('joint1','joint2','joint3','joint4','joint5','joint6')
            target_joint_states = self._get_joint_states_by_names(target_joint_names,
                                            ik_response.solution.joint_state.name, ik_response.solution.joint_state.position)
            return target_joint_states
        else:
            rospy.logerr(f"IK failed with error code: {ik_response.error_code.val}")
            return None

    def _get_joint_states_by_names(self, joint_names:tuple,all_joint_names:tuple,all_joint_values:tuple) -> list:
        joint_states = [0 for _ in range(len(joint_names))]
        for i, name in enumerate(joint_names):
            joint_states[i] = all_joint_values[all_joint_names.index(name)]
        return joint_states

    def register_callback(self,func,name):
        self.funcs[name] = func
