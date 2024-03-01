from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from ..base import Camera


class RosCamera(Camera, backend="ros"):

    def __init__(self, backend) -> None:
        super().__init__(backend)
        image_raw_topic_name = "/camera/color/image_raw"
        depth_topic_name = "/camera/aligned_depth_to_color/image_raw"
        rgb_camera_info_topic_name = "/camera/color/camera_info"
        self._rgb_img_suber = rospy.Subscriber(image_raw_topic_name,
                                               Image,
                                               queue_size=1,
                                               callback=self._rgb_img_call_back)
        self._depth_img_suber = rospy.Subscriber(
            depth_topic_name,
            Image,
            queue_size=1,
            callback=self._depth_img_call_back,
        )
        raw_rgb_image: Image = rospy.wait_for_message(image_raw_topic_name, Image, timeout=1)
        rgb_camera_info: CameraInfo = rospy.wait_for_message(rgb_camera_info_topic_name, CameraInfo, timeout=1)
        self._HEIGHT = raw_rgb_image.height
        self._WIDTH = raw_rgb_image.width
        if (rgb_camera_info.height != self._HEIGHT or rgb_camera_info.width != self._WIDTH):
            raise ValueError(
                f"rgb camera info and rgb image size not match, the image size is {[self._WIDTH,self._HEIGHT]}, the camera info is {rgb_camera_info.height,rgb_camera_info.width}"
            )
        self._INTRINSIC = np.array(rgb_camera_info.K).reshape((3, 3))
        rospy.wait_for_message(depth_topic_name, Image, timeout=1)

    @property
    def WIDTH(self) -> int:
        return self._WIDTH

    @property
    def HEIGHT(self) -> int:
        return self._HEIGHT

    @property
    def INTRINSIC(self) -> int:
        return self._INTRINSIC

    def init(self, *args, **kwargs) -> bool:
        self.inited = True
        return True

    def deinit(self) -> bool:
        self.inited = False
        return True

    def get_rgb(self) -> np.ndarray:
        return self._rgb_img

    def get_depth(self) -> np.ndarray:
        return self._depth_img

    def _rgb_img_call_back(self, msg: Image):
        self._rgb_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _depth_img_call_back(self, msg: Image):
        self._depth_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="32FC1")
