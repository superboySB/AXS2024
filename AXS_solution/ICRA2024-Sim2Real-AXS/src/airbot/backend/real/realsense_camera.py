import pyrealsense2 as rs
import numpy as np
import cv2
from numpy import size

from ..base import Camera


class RealsenseCamera(Camera, bakend='realsense'):

    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.backend = backend
        self.inited: bool = False
        # Image params: list[width,height,fps]
        self.img_config: list[int, int, int] = [1280, 720, 30]
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self._INTRINSIC = None
        self.inited = self.init()

    @property
    def WIDTH(self) -> int:
        return self.img_config[0]

    @property
    def HEIGHT(self) -> int:
        return self.img_config[1]

    @property
    def INTRINSIC(self) -> np.ndarray:
        return self._INTRINSIC

    def init(self, *args, **kwargs) -> bool:
        config = rs.config()
        config.enable_stream(rs.stream.depth,
                             self.img_config[0],
                             self.img_config[1],
                             rs.format.z16,
                             self.img_config[2])
        config.enable_stream(rs.stream.color,
                             self.img_config[0],
                             self.img_config[1],
                             rs.format.bgr8,
                             self.img_config[2])
        # Start streaming
        cfg = self.pipeline.start(config)
        i = 0
        while i < 50:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            i += 1
        profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for color stream
        intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
        # width, height, ppx, ppy, fx, fy, Brown_Conrady
        self._INTRINSIC = np.array([[intr.fx, 0, intr.ppx],
                                    [0, intr.fy, intr.ppy],
                                    [0, 0, 1]], dtype=np.dtypes.Float64DType)
        return True

    def deinit(self) -> bool:
        self.inited = False
        return True

    def get_rgb(self) -> np.ndarray:
        return self.__get_rgb_depth('rgb')

    def get_depth(self) -> np.ndarray:
        return self.__get_rgb_depth('depth')

    def __get_rgb_depth(self, img_type: str = 'both'):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rgb_color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if img_type == 'rgb':
            return rgb_color_image
        elif img_type == 'depth':
            return depth_image
        else:
            return color_image, depth_image


if __name__ == "__main__":
    backend = ''
    camera = RealsenseCamera(backend)
    rgb = camera.get_rgb()
    depth = camera.get_depth()
    print(size(rgb), type(rgb))
    print(size(depth), type(depth))
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', rgb)
    cv2.waitKey(1)
    cv2.imwrite(f"image_rgb.png", rgb)
