""" Predicting grasp pose from the input of point_cloud.
    Author: Hongrui-Zhu
    Note: 
        Pay attention to modifying camera parameters("self.camera_width"  "self.camera_high" "self.intrinsic" "self.factor_depth") to adapt to hardware
"""

import copy
import os
import numpy as np
import open3d as o3d

import torch
import yaml
from graspnetAPI import GraspGroup

from graspnet.models.graspnet import GraspNet as graspnet, pred_decode
from graspnet.utils.collision_detector import ModelFreeCollisionDetector
from graspnet.utils.quaternion_utils import rota_to_quat


class GraspPredictor:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, model, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[model] = cls

    def __new__(cls, model: str, **kwargs):
        subclass = cls._registry[model]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############

    def __init__(self, model, **kwargs) -> None:
        """Create GraspModel with specific model

        Args:
            model (str): type of model
        """
        self.model: str = model
        config_path = os.environ.get('LM_CONFIG')
        with open(config_path, 'r') as f:
            local_config = yaml.load(f, Loader=yaml.FullLoader)
        extra = local_config.get(model, {})
        self.kwargs = kwargs | extra
        self.ckpt_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')

    def infer(self, point_cloud: np.ndarray, workspace_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError
        return (translation, quaternion)  # translation[3], quaternion[4]


class GraspNet(GraspPredictor, model='graspnet'):

    def __init__(self, model) -> None:
        """
        Initialize the model and load checkpoints (if needed)
        """
        super().__init__(model)
        # load model
        self.checkpoint_path = os.path.join(self.ckpt_dir, self.kwargs['ckpt'])
        self.net = self.get_net()
        # net params
        self.num_point = 20000
        self.num_view = 300
        self.collision_thresh = 0.01
        self.voxel_size = 0.01

    # prompt: Optional[str] = None
    def infer(self, point_cloud: np.ndarray, workspace_mask) -> tuple[np.ndarray, np.ndarray]:
        """Obtain the target grasp pose given a point cloud
        Args:
            point_cloud: the point cloud array with shape (N, 3) and dtype np.float64
                Each of the N entry is the cartesian coordinates in the world base
            workspace_mask: mask for grasping targets

        Returns:
            tuple(np.ndarray, np.ndarray): the 6-DoF grasping pose in the world base 
                the first element: an np.ndarray with the size of [3] and dtype of np.float64, representing the target position of the end effector
                the second element: an np.ndarray with the size of [4] and dtype of np.float64, representing the orientation of the pose
                    this element is a quarternion representing the rotation between the target pose and (1,0,0) (pointing forward, the default orientation of end effector)

        Notes: the axis direction of the world base are:
            x -> forward
            y -> left
            z -> upward
        """
        net = self.get_net()
        end_points, cloud = self.get_and_process_data(point_cloud, workspace_mask)
        gg = self.get_grasps(net, end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        gg.nms()
        gg.sort_by_score()
        # self.vis_grasps(gg, cloud)
        translation = gg[0].translation
        quaternion = rota_to_quat(gg[0].rotation_matrix)
        vis = (gg,cloud)
        # translation[3], quaternion[4]
        return (translation, quaternion),vis

    def get_net(self):
        # Init the model
        net = graspnet(input_feature_dim=0,
                       num_angle=12,
                       num_depth=4,
                       cylinder_radius=0.05,
                       hmin=-0.02,
                       hmax_list=[0.01, 0.02, 0.03, 0.04],
                       is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self, input_cloud, workspace_mask):
        # generate cloud
        cloud = input_cloud[:, :, 0:3]
        color = input_cloud[:, :, 3:]
        # cloud_nomask = copy(cloud)
        # color_nomask = copy(color)
        cloud_nomask = cloud.reshape([-1, 3])
        color_nomask = color.reshape([-1, 3]) / 255.0

        # get valid points
        mask = workspace_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        cloud_masked = cloud_masked.reshape([-1, 3]).astype(np.float32)
        color_masked = color_masked.reshape([-1, 3]).astype(np.float32)

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_nomask.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_nomask.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud):
        gg = gg[:5]
        grippers = gg.to_open3d_geometry_list()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # Visualize the point cloud, grippers, and the coordinate frame
        o3d.visualization.draw_geometries([cloud, *grippers, coordinate_frame])



if __name__ == '__main__':
    pass
