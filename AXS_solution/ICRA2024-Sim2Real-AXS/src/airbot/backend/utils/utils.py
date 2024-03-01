import numpy as np
from scipy.spatial.transform import Rotation


def pose2mat(pose: tuple[np.ndarray, np.ndarray]):
    translation, quaternion = pose
    # build rotation matrix
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    # build trans matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def camera2base(camera_data, shift: tuple[np.ndarray, np.ndarray], pose: tuple):
    """
            pose : tuple(array([, , ]), array([,  ,  ,  ]))
        """
    t = (pose2mat(pose) @ pose2mat(shift))

    origin_shape = camera_data.shape
    if origin_shape[-1] > 3:
        feat = camera_data[..., 3:]
    camera_data = camera_data[..., :3].reshape(-1, 3)
    camera_data_homogeneous = np.hstack((camera_data, np.ones((camera_data.shape[0], 1))))
    base_data = (t @ camera_data_homogeneous.T).T[:, :3]
    base_data = base_data.reshape(*origin_shape[:-1], -1)
    if origin_shape[-1] > 3:
        base_data = np.concatenate([base_data, feat], axis=-1)
    return base_data

def armbase2world(arm_data, pose: tuple):
    """
            pose : tuple(array([, , ]), array([,  ,  ,  ]))
        """
    arm_data[0] += 0.2975        #armbase2carbase
    arm_data[1] += -0.17309
    arm_data[2] += 0.3488
    t = pose2mat(pose)
    origin_shape = arm_data.shape
    if origin_shape[-1] > 3:
        feat = arm_data[..., 3:]
    arm_data = arm_data[..., :3].reshape(-1, 3)
    camera_data_homogeneous = np.hstack((arm_data, np.ones((arm_data.shape[0], 1))))
    base_data = (t @ camera_data_homogeneous.T).T[:, :3]
    base_data = base_data.reshape(*origin_shape[:-1], -1)
    if origin_shape[-1] > 3:
        base_data = np.concatenate([base_data, feat], axis=-1)
    return base_data
