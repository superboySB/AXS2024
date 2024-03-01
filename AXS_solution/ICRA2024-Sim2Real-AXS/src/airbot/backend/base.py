from enum import Enum

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation
"""
Unless otherwise specified, all methods are **blocking** synchronous methods.
"""


class Gripper:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, backend, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[backend] = cls

    def __new__(cls, backend: str):
        subclass = cls._registry[backend]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############
    """The base class of AirBot gripper backend
    FIXME currently the actual grasping pose is managed by Arm. This should be fixed in the future.
    """

    class Status(Enum):
        OPEN = 0
        CLOSE = 1
        MOVING = 2
        ERROR = 3

    def __init__(self, backend: str) -> None:
        """Create Gripper with specific backend

        Args:
            backend (str): type of backend
        """
        self.backend: str = backend
        self.inited: bool = False

    def init(self, *args, **kwargs) -> bool:
        """Initialize the backend of AirBot gripper
        After initialization, self.inited should be True

        Returns:
            bool: True if initialize successful, False otherwise
        """
        raise NotImplementedError

    def deinit(self) -> bool:
        """Deinitialize the backend of AirBot gripper
        After uninitialization, self.inited should be False

        Returns:
            bool: True if uninitialize successful, False otherwise
        """
        raise NotImplementedError

    @property
    def state(self) -> Status:
        """The current state of the gripper

        Returns:
            Status: The state of the gripper. Valid options:
            - Gripper.Status.OPEN
            - Gripper.Status.CLOSE
            - Gripper.Status.ERROR
        """
        raise NotImplementedError

    def open(self) -> Status:
        """Open the gripper

        Returns:
            Status: The state of the gripper after action. Valid options:
            - Gripper.Status.OPEN
            - Gripper.Status.CLOSE
            - Gripper.Status.ERROR
        """
        raise NotImplementedError

    def close(self) -> Status:
        """Close the gripper

        Returns:
            Status: The state of the gripper after action. Valid options:
            - Gripper.Status.OPEN
            - Gripper.Status.CLOSE
            - Gripper.Status.ERROR
        """
        raise NotImplementedError


class Camera:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, backend, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[backend] = cls

    def __new__(cls, backend: str):
        subclass = cls._registry[backend]
        obj = object.__new__(subclass)
        return obj

    def __init__(self, backend) -> None:
        self.backend = backend
        self.inited: bool = False

    ######### Factory method end ############

    @property
    def WIDTH() -> int:
        """The horizontal width (y-axis) of RGB and depth camera in pixels

        Returns:
            int: width of the camera
        """
        raise NotImplementedError

    @property
    def HEIGHT() -> int:
        """The vertical height (x-axis) of RGB and depth camera in pixels

        Returns:
            int: height of the camera
        """
        raise NotImplementedError

    @property
    def INTRINSIC() -> np.ndarray:
        """The 3x3 intrinsic parameter of the cameras.

        Returns:
            np.ndarray: with shape of (3, 3) and dtype of np.float64
        """
        raise NotImplementedError

    def init(self, *args, **kwargs) -> bool:
        """Initialize the backend of AirBot camera
        After initialization, self.inited should be True

        Returns:
            bool: True if initialize successful, False otherwise
        """
        raise NotImplementedError

    def deinit(self) -> bool:
        """
        Uninitialize the camera.
        After uninitialization, self.inited should be False

        Returns:
            bool: True if uninitialize successful, False otherwise
        """
        raise NotImplementedError

    def get_rgb(self) -> np.ndarray:
        """Capture and return the color image in RGB format

        Returns:
            np.ndarray: with shape of (self.HEIGHT, self.WIDTH, 3) and dtype of np.uint8
        """

        raise NotImplementedError

    def get_depth(self) -> np.ndarray:
        """Capture and return the depth image in meters
        depth is the distance projected on the z-axis of the camera coordinate system

        Returns:
            np.ndarray: with shape of (self.HEIGHT, self.WIDTH) and dtype of np.float32
        """
        raise NotImplementedError


class Arm:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, backend, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[backend] = cls

    def __new__(cls, backend: str):
        subclass = cls._registry[backend]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############
    """The base class of AirBot arm backend
    The base link for arm has x axis pointing forward, y axis pointing left, z axis pointing up
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.inited: bool = False

    def init(self, *args, **kwargs) -> bool:
        """Initialize the backend of AirBot arm
        After initialization, self.inited should be True

        Returns:
            bool: True if initialize successful, False otherwise
        """
        raise NotImplementedError

    def deinit(self, *args, **kwargs) -> bool:
        """Uninitialize the backend of AirBot arm
        After uninitialization, self.inited should be False

        Returns:
            bool: True if uninitialize successful, False otherwise
        """
        raise NotImplementedError

    @property
    def INIT_END_POSE(self) -> Tuple[np.ndarray, np.ndarray]:
        """The default 6-DoF pose of the arm end when the arm is initialized
        The pose is relative to the arm base
        NOTE currently by "end" we mean the end of the gripper (the grasping pose), not the end of the arm. This should be fixed in the future.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                cartesian position (with shape of (3, ) and dtype of np.float64) 
                quaternion rotation (with shape of (4, ) and dtype of np.float64)
        """
        raise NotImplementedError

    @property
    def end_pose(self) -> tuple:
        """Return current 6-DoF pose of arm end relative to the arm base
        NOTE currently by "end" we mean the end of the gripper (the grasping pose), not the end of the arm. This should be fixed in the future.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                cartesian position (with shape of (3, ) and dtype of np.float64) 
                quaternion rotation (with shape of (4, ) and dtype of np.float64)
        """
        raise NotImplementedError

    def move_end_to_pose(self, position: np.ndarray, rotation: np.ndarray) -> tuple:
        """Move the end effector to a given 6-DoF pose relative to the arm base
        Return the actual 6-DoF pose of the end effector after action
        NOTE currently by "end" we mean the end of the gripper (the grasping pose), not the end of the arm. This should be fixed in the future.

        Raises:
            ValueError: if the given pose is not reachable. The arm should not move if this error is raised

        Args:
            position (np.ndarray): cartesian position (with shape of (3, ) and dtype of np.float64)
            rotation (np.ndarray): quaternion rotation (with shape of (4, ) and dtype of np.float64) 

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                cartesian position (with shape of (3, ) and dtype of np.float64) 
                quaternion rotation (with shape of (4, ) and dtype of np.float64)
        """
        raise NotImplementedError


class Base:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, backend, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[backend] = cls

    def __new__(cls, backend: str):
        subclass = cls._registry[backend]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############

    @property
    def INIT_POSITION(self) -> np.ndarray:
        """The cartesian position of the base when the base is initialized
        the position is relative to the world coordinate system

        Returns:
            np.ndarray: with shape of (3, ) and dtype of np.float64
        """
        raise NotImplementedError

    @property
    def INIT_ROTATION(self) -> np.ndarray:
        """The quaternion rotation of the base when the base is initialized
        the rotation is relative to the world coordinate system

        Returns:
            np.ndarray: with shape of (4, ) and dtype of np.float64
        """
        raise NotImplementedError

    def __init__(self, backend) -> None:
        self.inited: bool = False
        self.backend = backend

    def init(self, *args, **kwargs) -> bool:
        """Initialize the backend of AirBot base
        After initialization, self.inited should be True

        Returns:
            bool: True if initialize successful, False otherwise
        """
        raise NotImplementedError

    def deinit(self, *args, **kwargs) -> bool:
        """Initialize the backend of AirBot base
        After initialization, self.inited should be True

        Returns:
            bool: True if initialize successful, False otherwise
        """
        raise NotImplementedError

    @property
    def position(self) -> np.ndarray:
        """Get the current position of the base in the world coord system

        Returns:
            np.ndarray: with shape of (3, ) and dtype of np.float64
        """
        raise NotImplementedError

    @property
    def rotation(self) -> np.ndarray:
        """Get the current rotation of the base in the world coord system

        Returns:
            np.ndarray: with shape of (4, ) and dtype of np.float64
        """
        raise NotImplementedError

    def move_to(self, position: np.ndarray, rotation: np.ndarray) -> tuple:
        """Move the base to a given position and rotation in the world coord system

        Raises:
            ValueError: if the given pose is not reachable. The base should not move if this error is raised

        Args:
            position (np.ndarray): with shape of (3, ) and dtype of np.float64
            rotation (np.ndarray): with shape of (4, ) and dtype of np.float64

        Returns:
            tuple: self.position, self.rotation
        """
        raise NotImplementedError


class AirBot:

    def __init__(
        self,
        arm_backend='mock',
        base_backend='mock',
        camera_backend='mock',
        gripper_backend='mock',
    ) -> None:
        self.arm = Arm(backend=arm_backend)
        self.base = Base(backend=base_backend)
        self.camera = Camera(backend=camera_backend)
        self.gripper = Gripper(backend=gripper_backend)

    def init(self, *args, **kwargs) -> bool:
        """
        """
        self.arm.init()
        self.camera.init()
        self.gripper.init()

    def deinit(self) -> bool:
        """
        """
        self.arm.deinit()
        self.camera.deinit()
        self.gripper.deinit()
