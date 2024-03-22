# %%
import warnings

warnings.filterwarnings("ignore")
from airbot.backend import Arm, Camera, Base, Gripper
import os
import numpy as np
import copy
from airbot.backend.utils.utils import camera2base, armbase2world

from airbot.grasp.graspmodel import GraspPredictor
from PIL import Image
import time
import cv2
from airbot.example.utils.draw import draw_bbox, obb2poly
from airbot.example.utils.vis_depth import vis_image_and_depth
from scipy.spatial.transform import Rotation
from threading import Thread, Lock
import math
import torch
import yaml
import sys
import shutil
from typing import Any,Union,List
from datetime import datetime

from airbot.lm.utils import depth2cloud

os.environ['LM_CONFIG'] = "/root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/local.yaml"
os.environ['CKPT_DIR'] = '/root/Workspace/AXS_baseline/ckpt'

import logging
import time
# 创建一个logger
logger = logging.getLogger('bit-linc')
logger.setLevel(logging.INFO)  # 设置日志级别

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler('/root/Workspace/AXS_baseline/app.log')
file_handler.setLevel(logging.INFO)

# 再创建一个handler，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class Solution:

    CAMERA_SHIFT = np.array([-0.093, 0, 0.07]), np.array([0.5, -0.5, 0.5, -0.5])

    BEFORE_MW_BASE_POSE = (np.array([
        -0.02596185755302699,
        -0.5065354084074988,
        -0.17365354084074988,
    ]), np.array([
        0.00179501,
        0.06667202,
        0.04863613,
        0.99658725,
    ]))

    ARM_POSE_TO_MICROWAVE = (np.array([0.67335, 0.2, -0.07678]), np.array([0.0, 0.0, 0.0, 1.0]))
 
    POSE_CLOSE_MICROWAVE = (np.array([-0.05931371154832413, 0.10621326743872146, 0.052710225767309826]), np.array([0.00476639,  0.08115882, 0.05789078, 0.99500713]))

    ARM_POSE_CLOSE_MICROWAVE = (np.array([0.47335, 0.1, 0.05]), np.array([0.0, 0.0, 0.0, 1.0]))

    ARM_POSE_CLOSE_MICROWAVE_END = (np.array([0.4, -0.55, 0.05]), np.array([0.0, 0.0, 0.0, 1.0]))

    POSE_TO_BOWL = (np.array([-0.01226429675289562, 0.11259263609492792, -0.042619463119529605]),
                    np.array([-0.05418989389554988, 0.056031992518506414, 0.707, 0.707]))
    
    ARM_POSE_TO_LOWER_CABINET = (np.array([0.62335, 0.25, -0.05678]), np.array([0.0, 0.0, 0.0, 1.0]))

    ARM_POSE_PUT_LOWER = (np.array([0.62335, 0.25, -0.09678]), np.array([0.0, 0.0, 0.0, 1.0]))

    ARM_POSE_TO_UPPER_CABINET = (np.array([0.62335, 0.25, 0.22]), np.array([0.0, 0.0, 0.0, 1.0]))

    ARM_POSE_PUT_UPPER = (np.array([0.62335, 0.25, 0.17]), np.array([0.0, 0.0, 0.0, 1.0]))

    POSE_OPEN_CAB = (np.array([-0.2741637241544763, 0.3241416117180577, -0.07743623649227918]), np.array([-0.026456952502619244, 0.022510511678367467, 0.6642090190392874, 0.7467393692280592]))
    
    ARM_POSE_STANDARD_MOVING = (np.array([0.3225, 0.00, 0.219]), np.array([0.0, 0.0, 0.0, 1.0]))

    GRASP_POSE_1 = (np.array([0.345502073568463, 0.49365995419306763, 0.07947950001408821]), np.array(
        [-0.0051582, 0.09461529, 0.05903872, 0.99374834]))

    GRASP_POSE_2 = (np.array([0.40524870062559315, 0.4479303912730242, 0.004671858854359251]), np.array(
        [-0.04836788, 0.0417043, 0.66597635, 0.74323402]))

    OBSERVE_ARM_POSE_1 = (np.array([
        0.2835699,
        0.2,
        0.171663168,
    ]), np.array([
        -0.13970062182177911,
        0.6487791800204252,
        0.032918235938941776,
        0.7473190092439113,
    ]))
    
    OBSERVE_ARM_POSE_2 = (np.array([
        0.2835699,
        -0.2,
        0.171663168,
    ]), np.array([
        -0.13970062182177911,
        0.6487791800204252,
        0.032918235938941776,
        0.7473190092439113,
    ]))

    END_POSE = (np.array([-1.2317611908186263, -0.9177336410440479, -0.2374505629662929]), 
                np.array([0.006592923324957343, -0.012942241749323641, 0.014944697147015459, 0.9997828203003477]))

    def __init__(self):
        self.arm = Arm(backend='ros')

        self.base = Base(backend='ros')

        self.gripper = Gripper(backend='ros')
        self.gripper.open()

        self.camera = Camera(backend='ros')

        # self.detector = Detector(model='grounding-dino')
        # self.detector = Detector(model='yolo-v7')
        # self.detector = Detector(model='yolo-world')
        from ultralytics import YOLO
        self.detector = YOLO("/root/Workspace/YOLOv8-TensorRT/yolov8l-world.pt")
        self.detector.set_classes(["brown cab"])

        # self.segmentor = SegmentAnything(model='segment-anything')
        sys.path.append("/root/Workspace/efficientvit/")
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        from efficientvit.sam_model_zoo import create_sam_model

        efficientvit_sam = create_sam_model("xl1", True, "/root/Workspace/efficientvit/assets/checkpoints/sam/xl1.pt")
        efficientvit_sam = efficientvit_sam.cuda(0).eval()
        self.segmentor = EfficientViTSamPredictor(efficientvit_sam)


        # TODO: cuda kernel error in 12.2/12.1
        self.grasper = GraspPredictor(model='graspnet')

        self.image_lock = Lock()
        self.result_lock = Lock()
        self.prompt_lock = Lock()
        self.running = True
        self.prompt = 'sky'
        self.update_once()

        # 实例分割、定位的结果可视化后台执行
        self.t_vis = Thread(target=self.vis, daemon=True)
        self.t_vis.start()
        
        # 实例分割、定位的算法后台执行
        self.t_update = Thread(target=self.update, daemon=True)
        self.t_update.start()

    @property
    def image(self):
        with self.image_lock:
            return copy.deepcopy(self._image)

    @image.setter
    def image(self, value):
        with self.image_lock:
            self._image = copy.deepcopy(value)

    @property
    def prompt(self):
        with self.prompt_lock:
            return copy.deepcopy(self._prompt)

    @prompt.setter
    def prompt(self, value):
        with self.prompt_lock:
            self._prompt = copy.deepcopy(value)

    @property
    def depth(self):
        with self.image_lock:
            return copy.deepcopy(self._depth)

    @depth.setter
    def depth(self, value):
        with self.image_lock:
            self._depth = copy.deepcopy(value)

    @property
    def bbox(self):
        with self.result_lock:
            return copy.deepcopy(self._bbox)

    @bbox.setter
    def bbox(self, value):
        with self.result_lock:
            self._bbox = copy.deepcopy(value)

    @property
    def mask(self):
        with self.result_lock:
            return copy.deepcopy(self._mask)

    @mask.setter
    def mask(self, value):
        with self.result_lock:
            self._mask = copy.deepcopy(value)

    # 获取一次实例分割结果
    def update_once(self):
        with self.image_lock, self.result_lock:
            self._image = copy.deepcopy(self.camera.get_rgb())
            self._depth = copy.deepcopy(self.camera.get_depth())

            # self._det_result = self.detector.infer(self._image, self._prompt)
            raw_det_result = self.detector.predict(self._image)
            # 简单的NMS
            if raw_det_result[0].boxes.data.shape[0] > 0:
                max_value_index = torch.argmax(raw_det_result[0].boxes.conf)
                x1,y1,x2,y2 = raw_det_result[0].boxes.xyxy[max_value_index].cpu().numpy()
                self._det_result = {
                    "bbox": torch.tensor([(y1+y2)/2, (x1+x2)/2, y2-y1, x2-x1]),  
                    "score": raw_det_result[0].boxes.conf[max_value_index], 
                }
            else:
                self._det_result = {"bbox": torch.tensor(np.array([0., 0., 0., 0.])), "score": 0.}

            self._bbox = self._det_result['bbox'].cpu().numpy().astype(int)
            self.segmentor.set_image(self._image)
            width = self._image.shape[1]
            height = self._image.shape[0]
            gapped_bbox = np.zeros(4)
            gapped_bbox[0] = np.maximum(0, self._bbox[1]- self._bbox[3]/2- 15)
            gapped_bbox[1] = np.maximum(0, self._bbox[0]- self._bbox[2]/2 - 15)
            gapped_bbox[2] = np.minimum(width, self._bbox[1] + self._bbox[3]/2 + 15)
            gapped_bbox[3] = np.minimum(height, self._bbox[0] + self._bbox[2]/2 + 15)
            masks, scores, _ = self.segmentor.predict(
                point_coords= np.array([[self._bbox[1],self._bbox[0]]]),
                point_labels=np.array([1]),
                box = gapped_bbox,
                multimask_output=False,
            )
            
            max_idx = np.argmax(scores)

            self._sam_result =  {
                "mask": masks[max_idx].astype(bool),  # np.ndarray
                "score": scores[max_idx],  # str
            }

            # self._sam_result = self.segmentor.infer(self._image, self._bbox[None, :2][:, [1, 0]])

            self._mask = self._sam_result['mask']

    def update(self):
        while self.running:
            self.update_once()
            time.sleep(0.005)

    def vis(self):
        # 检查目标目录
        log_dir = "log"
        if os.path.exists(log_dir):
            # 清空目录
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            # 无限循环以显示图像
            while self.running:
                image_draw = self.image
                image_draw = image_draw * (self.mask[:, :, None].astype(np.uint8) * 0.75 + 0.25)
                image_draw = draw_bbox(image_draw, obb2poly(self.bbox[None, ...]).astype(int))
                image_draw = image_draw.astype(np.uint8)
                image_show = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)
                # cv2.imshow('RGB', image_show)
                _flag = np.any(self.mask)
                if _flag == True:
                    cv2.putText(image_show,
                                f"det score: {self._det_result['score']}, sam score: {self._sam_result['score']}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.imshow('RGB', image_show)
                
                # 生成包含毫秒的唯一文件名
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # 去掉最后三位以得到毫秒
                save_path = os.path.join(log_dir, f"{timestamp}.png")
                # 保存图像
                cv2.imwrite(save_path, image_show)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Exiting due to user interruption.")
        finally:
            cv2.destroyAllWindows()

    @staticmethod
    def _bbox2mask(image, bbox):
        mask = np.zeros_like(image[:, :, 0], dtype=bool)
        mask[
            bbox[0] - bbox[2] // 2:bbox[0] + bbox[2] // 2,
            bbox[1] - bbox[3] // 2:bbox[1] + bbox[3] // 2,
        ] = True
        return mask

    @staticmethod
    def base_cloud(image, depth, intrinsic, shift, end_pose):
        cam_cloud = depth2cloud(depth, intrinsic)
        cam_cloud = np.copy(np.concatenate((cam_cloud, image), axis=2))
        return camera2base(cam_cloud, shift, end_pose)
    
    # TODO: 抓取本身不容易失败，但是定位有问题的时候很容易对着空气抓
    def grasp(self):
        method = "2"

        # 使用with self.image_lock, self.result_lock:获取图像、深度信息、边界框(bbox)和掩码(mask)的副本，以防止在抓取过程中数据被其他线程修改。
        with self.image_lock, self.result_lock:
            _depth = copy.deepcopy(self._depth)
            _image = copy.deepcopy(self._image)
            _bbox = copy.deepcopy(self._bbox)
            _mask = copy.deepcopy(self._mask)

        # 计算抓取位置和旋转：
        # 使用self.base_cloud方法计算出基于相机视角的点云数据。
        cloud = self.base_cloud(_image, _depth, self.camera.INTRINSIC, self.CAMERA_SHIFT, self.arm.end_pose)

        # 根据method的值选择不同的抓取策略。这里method == "2"，所以直接计算出抓取位置grasp_position和旋转grasp_rotation。
        # 抓取位置基于目标的边界框bbox，旋转则是根据预设角度来确定。
        if method == "1":
            direction = cloud[_bbox[0] - _bbox[2] // 2, _bbox[1]][:3] - self.arm.end_pose[0]
            direction = direction / np.linalg.norm(direction)
            grasp_position = cloud[_bbox[0] - _bbox[2] // 2 + 9, _bbox[1]][:3] - 0.12 * direction
            grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, np.pi / 2], degrees=False).as_quat()
        elif method == "2":
            grasp_position = cloud[ _bbox[0], _bbox[1] - _bbox[3] // 2 + 8][:3]
            grasp_position[2] = -0.178
            grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, 0], degrees=False).as_quat()
        else:
            bbox_mask = self._bbox2mask(_image, _bbox)
            (grasp_position, grasp_rotation), _ = self.grasper.infer(cloud, bbox_mask)

            grasp_rotation = Rotation.from_euler("yz", [np.pi / 2, np.pi / 2], degrees=False).as_quat()

        # 执行抓取动作：首先，机械臂移动到计算出的抓取位置和旋转角度。等待2秒后，夹爪闭合，尝试抓取目标物体。
        # 之后，再次等待4秒，确保夹爪稳固抓取后，将机械臂移动回标准移动位置。
        self.arm.move_end_to_pose(grasp_position, grasp_rotation)
        time.sleep(4)
        self.gripper.close()
        time.sleep(5)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

    # 这是一个静态方法，用于可视化抓取过程。它不直接参与机器人的物理操作，而是用于调试和展示抓取策略。
    # 准备点云数据：使用open3d库创建一个点云对象，并填充点云和颜色信息。
    # 创建抓取组：使用GraspGroup类创建一个包含抓取位置和旋转的抓取表示。
    # 可视化：显示点云、抓取位置和世界坐标系，以便于理解抓取动作在空间中的具体表现。
    @staticmethod
    def _vis_grasp(cloud, position, rotation):
        import open3d as o3d
        from graspnetAPI.grasp import GraspGroup
        o3d_cloud = o3d.geometry.PointCloud()
        cloud = copy.deepcopy(cloud)
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud[:, :, :3].reshape(-1, 3).astype(np.float32))
        o3d_cloud.colors = o3d.utility.Vector3dVector(cloud[:, :, 3:].reshape(-1, 3).astype(np.float32) / 255.)
        gg = GraspGroup(
            np.array([1., 0.06, 0.01, 0.06, *Rotation.from_quat(rotation).as_matrix().flatten(), *position,
                      0]).reshape(1, -1))
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([o3d_cloud, *gg.to_open3d_geometry_list(), coordinate_frame])

    # 这个方法处理将抓取到的物体放入微波炉中的逻辑。
    # 移动到放置位置：首先，移动机器人底座到微波炉前的特定位置。
    # 移动机械臂到微波炉门口：然后，将机械臂移动到微波炉门口的位置，准备放置物体。
    # 打开夹爪释放物体：通过打开夹爪来释放物体。
    # 调整底座位置：最后，调整机器人底座的位置，以完成放置过程。
    def place_microwave(self):
        self.base.move_to(*self.BEFORE_MW_BASE_POSE, 'world', False)
        time.sleep(3)

        self.arm.move_end_to_pose(*self.ARM_POSE_TO_MICROWAVE)
        input_pose = (np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*input_pose, 'robot', True)
        self.gripper.open()
        time.sleep(2)
        output_pose = (np.array([-0.3, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*output_pose, 'robot', True)
    
    # 这个方法用于关闭微波炉门。
    # 移动机械臂到标准位置：首先，确保机械臂处于标准移动位置，避免在移动底座时发生碰撞。
    # 移动底座到关闭门的位置：将底座移动到可以关闭微波炉门的位置。
    # 执行关闭门的动作：通过机械臂执行一系列动作，推动微波炉门关闭。
    # 返回标准位置：完成操作后，机械臂移回标准位置，准备执行下一步操作。
    def close_microwave(self):
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)
        self.base.move_to(*self.POSE_CLOSE_MICROWAVE, 'world', False)
        time.sleep(2)

        self.arm.move_end_to_pose(*self.ARM_POSE_CLOSE_MICROWAVE)
        self.arm.move_end_to_pose(*self.ARM_POSE_CLOSE_MICROWAVE_END)
        output_pose1 = (np.array([-0.25, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*output_pose1, 'robot', True)

        time.sleep(2)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

    # 将碗放置到低柜
    def place_bowl_lower(self):
        # 使机械臂先移动到一个预设的标准位置，这个位置通常是安全的起始点或中转点，用来避免在移动过程中的碰撞。
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)
        # 将机器人的底座移动到便于摆放碗的位置，这里的位置是相对于世界坐标系的。
        self.base.move_to(*self.POSE_TO_BOWL, 'world', False)

        # 调整机械臂到低柜的放置位置，准备将碗放入。
        self.arm.move_end_to_pose(*self.ARM_POSE_TO_LOWER_CABINET)
        # 进行微调，确保机械臂能够精确放置碗。这个位置是相对于机器人坐标系的，意味着是基于机器人当前位置的相对移动。
        input_pose = (np.array([0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.base.move_to(*input_pose, 'robot', True)
        # 打开夹爪，释放碗。同时给夹爪一个向下的力，确保平稳。
        self.gripper.open()
        self.arm.move_end_to_pose(*self.ARM_POSE_PUT_LOWER)
        time.sleep(2)
        output_pose = (np.array([- 0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        # 将机械臂和底座分别移回到初始的或安全的位置，为下一个操作做准备。
        self.base.move_to(*output_pose, 'robot', True)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

    # 将碗放置到高柜
    # 此函数的操作与place_bowl_lower非常相似，主要区别在于放置位置的高度不同。
    def place_bowl_upper(self):
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)
        self.base.move_to(*self.POSE_TO_BOWL,'world', False)

        self.arm.move_end_to_pose(*self.ARM_POSE_TO_UPPER_CABINET)
        input_pose = (np.array([0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.base.move_to(*input_pose, 'robot', True)
        self.gripper.open()
        self.arm.move_end_to_pose(*self.ARM_POSE_PUT_UPPER)
        time.sleep(2)
        output_pose = (np.array([-0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.base.move_to(*output_pose, 'robot', True)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

    # 利用大模型做实例分割之后、开始点云转换、输出物体中心相对世界坐标系下位置的核心函数
    def lookforonce(self, det_th, sam_th):
        with self.image_lock, self.result_lock:
            _rgb = copy.deepcopy(self.camera.get_rgb())
            _depth = copy.deepcopy(self.camera.get_depth())
            _det_result = copy.deepcopy(self._det_result)
            _sam_result = copy.deepcopy(self._sam_result)
        _bbox = _det_result['bbox'].cpu().numpy().astype(int)  # 检测框
        _mask = _sam_result['mask'] # 掩码结果
        if np.any(_mask) is False: # 检查分割掩码是否存在。如果掩码完全不存在（即没有找到任何目标物体），则打印提示信息并跳过后续步骤。
            logger.info(f"direction {direction} not Found")
            
        logger.info(f"det score: {_det_result['score']}")
        logger.info(f"sam score: {_sam_result['score']}")
        if _det_result['score'] > det_th and _sam_result['score'] > sam_th:
            logger.info(f"Found the {self._prompt}")
            # 接下来这段代码主要执行的是将一个物体在图像中的中心点位置转换到世界坐标系下的位置。
            # 第一步：将深度图转换为点云（depth2cloud）
            # 这一步使用深度图和相机内参（self.camera.INTRINSIC），将图像中的像素点转换为相机坐标系下的三维点云。点云中的每个点表示相对于相机位置的三维空间点。
            # depth2cloud(_depth, self.camera.INTRINSIC, organized=True)：这个函数根据给定的深度图和相机的内参矩阵，计算出对应的三维点云。organized=True参数意味着输出的点云保持与输入图像相同的行和列结构，方便后续处理。
            # [_bbox[0] // 1, _bbox[1] // 1]：从计算得到的点云中，根据目标物体的边界框（BoundingBox，简称BBox），选取中心点的三维坐标。这里的中心点坐标是通过边界框来确定的，用于表示目标物体的大致位置。
            centerpoint = depth2cloud(_depth, self.camera.INTRINSIC, organized=True)[_bbox[0] // 1, _bbox[1] // 1]
            
            # 第二步：从相机坐标系转换到基座坐标系（camera2base）
            # 这一步是将上一步得到的点（在相机坐标系下）转换到机器人基座的坐标系中。这个转换考虑了相机相对于机器人基座的位置和姿态。
            # camera2base(centerpoint, self.CAMERA_SHIFT, self.arm.end_pose)：此函数接收三个参数：第一个参数是从相机坐标系中得到的点，第二个参数self.CAMERA_SHIFT是相机相对于机器人基座的平移向量，第三个参数self.arm.end_pose是机械臂末端的位置和姿态。
            # 这个函数计算得到的是，点在机器人基座坐标系中的位置。
            centerpoint = camera2base(centerpoint, self.CAMERA_SHIFT, self.arm.end_pose)
            
            # 第三步：从基座坐标系转换到世界坐标系（armbase2world）
            # 这一步进一步将上一步得到的基座坐标系下的点转换到世界坐标系中。这主要涉及到机器人底座在世界坐标系中的位置和姿态。
            # 此函数将基座坐标系下的点转换到世界坐标系下。它需要机器人底座的位置self.base.position和旋转self.base.rotation（即机器人底座在世界坐标系中的位置和姿态）。
            centerpoint = (armbase2world(centerpoint, (self.base.position, self.base.rotation)).squeeze())
            
            # 第四步：根据目标物体的边界框在RGB图像中截取目标区域，然后计算这个区域内所有像素的平均RGB颜色值。
            object_rgb = _rgb[_bbox[0] - np.int32(_bbox[2]/4):_bbox[0] + np.int32(_bbox[2]/4), _bbox[1] - np.int32(_bbox[3]/4):_bbox[1] + np.int32(_bbox[3]/4)]
            mean_rgb = (np.mean(np.mean(object_rgb, axis=0), axis=0).astype(int))
            
            logger.info(f"-----------------------------------------------------")
            logger.info(f"centerpoint is {centerpoint}")
            logger.info(f"object rgb is {mean_rgb}")
            logger.info(f"-----------------------------------------------------")

            return centerpoint, mean_rgb

if __name__ == '__main__':
    s = Solution()

    # -----------------------------------------------------------------
    # 任务一需要保证能够优先、稳定完成
    # 计划：寻找并抓取白色的杯子，并放到微波炉中，关闭微波炉门
    logger.info("First, I plan to find white mug")
    cup_prompts = ['white cup with a handle','white mug','white cup']
    s.detector.set_classes(["A white cup with a handle"])
    s.base.move_to(*s.GRASP_POSE_1, 'world', False)  # TODO: 这些local planner的nav效果有点蠢,多试试看
    time.sleep(2)
    cp = None
    look_num = 0
    # 循环寻找目标物体
    # 这个循环通过改变direction值，控制机械臂移动到两个不同的观察位置（OBSERVE_ARM_POSE_1和OBSERVE_ARM_POSE_2），分别对应两个方向。
    # 在每个观察位置，调用lookforonce方法尝试检测目标物体。这个方法接受两个参数（0.65, 0.65），分别是检测阈值和分割阈值，用于评估检测和分割结果的可信度。
    # 如果找到符合条件的物体，lookforonce将返回中心点centerpoint和物体平均RGB颜色值object_mean_rgb，否则返回None。
    # 如果连续三次（look_num > 3）循环都没有找到目标物体，则退出循环，意味着未能找到目标。
    while cp is None:
        
        for direction in [1, 2]:
            if direction == 1:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_1)
            else:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_2)
            time.sleep(3)
            cp = s.lookforonce(0.65,0.65)
            if cp is not None:
                break
        look_num += 1
        if look_num>2:
            break
    
    if cp is None:
        s.base.move_to(*s.GRASP_POSE_2, 'world', False) 
        
        for direction in [1, 2]:
            if direction == 1:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_1)
            else:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_2)
            time.sleep(3)
            cp = s.lookforonce(0.65,0.65)
            if cp is not None:
                break

    # 对找到的目标进行操作
    if cp is not None:
        centerpoint, object_mean_rgb = cp    
        # 计算物体相对于机器人底座的位置。这个计算涉及到从世界坐标系到机器人底座坐标系的转换，确保机械臂能够准确地抓取到物体。
        centerp_car = np.linalg.inv(np.array(Rotation.from_quat(s.base.rotation).as_matrix())).dot((centerpoint-s.base.position))

        # 对齐坐标系，计算一个新的机械臂位置OBSERVE_ARM_POSE_TOP，用于从上方观察并准备抓取物体。
        # 注意arm_base 相对 car_base_link 的 offset是 (0.2975, -0.17309, 0.3488)
        # TODO: 这里没有用到6-DOF，而是固定了高度和抓取姿态（基本是从正上方抓取）
        OBSERVE_ARM_POSE_TOP = (np.array([
                    centerp_car[0]- 0.2975 - 0.05,
                    centerp_car[1] + 0.17309,
                    0.018713334665877806,
                ]), np.array([
                    -0.13970062182177911,
                    0.6487791800204252,
                    0.032918235938941776,
                    0.7473190092439113,
                ]))
        
        # 移动机械臂到上方观察/抓取位置。
        s.arm.move_end_to_pose(*OBSERVE_ARM_POSE_TOP)
        time.sleep(1)

        # 执行抓取动作。这个方法内部将处理抓取的具体逻辑，包括机械臂的精确移动和夹爪的控制。
        s.grasp()
        time.sleep(2)

        # 把抓取到的物体放置到微波炉中。
        logger.info("I plan to place_microwave")
        s.place_microwave()
        time.sleep(3)
    
    # 关闭微波炉门。
    logger.info("I plan to close_microwave")
    s.close_microwave()
    time.sleep(6)

    # --------------------------------------------------------------------------------
    # TODO: 比较难的任务，目前最重要的是能够智能检测柜门是否打开（利用realsense）
    # 将机器人的底座移动到打开柜门的起始位置和姿态。这个位姿是相对于世界坐标系的，用于机器人接近柜门以便后续打开它。
    for _ in range(2):
        logger.info("Now I try to open the cab")
        s.base.move_to(*s.POSE_OPEN_CAB, 'world', False) 
        time.sleep(2)

        # 表示柜门把手在世界坐标系中的位置。这个位置是预先测量或通过某种方式计算得出的。
        POS_DOOR_HANDLE = np.array([0.30353946626186371, 1.230472918510437, 0])

        # 计算底座旋转矩阵的逆，然后通过点乘操作将上一步计算的向量转换到底座的局部坐标系中。
        # 这一步是为了得到把手位置相对于机器人底座的局部坐标，因为接下来的s.arm.move_end_to_pose操作是基于机器人本身的坐标系进行的。
        centerp_car = np.linalg.inv(np.array(Rotation.from_quat(s.base.rotation).as_matrix())).dot((POS_DOOR_HANDLE-s.base.position))
        
        # 确定机械臂的目标位置和姿态
        # 位置是根据把手的局部坐标计算的。这里的数值调整是基于机器人的具体尺寸和抓取动作的需要。
        # 注意arm_base 相对 car_base_link 的 offset是 (0.2975, -0.17309, 0.3488)
        # 姿态(np.array([0.0, 0.0, 0.0, 1.0]))这里是一个四元数，表示机械臂末端执行器的朝向。
        # 在这个例子中，它被设置为没有旋转（即朝向不变），这意味着末端执行器将保持默认的方向不变。
        ARM_POSE_DOOR_HANDLE = (np.array([
                        centerp_car[0] - 0.2975 -0.01,
                        centerp_car[1] + 0.17309 -0.01,
                        0.2,  # 夹爪相对起点的末端高度
                    ]), np.array([
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]))

        # 控制机械臂移动到柜门把手的位置, 并使用gripper.close()命令闭合夹爪来抓住把手。（基于arm_base坐标系）
        logger.info("Arrived the cap and then i plan to grip the cab")      
        s.arm.move_end_to_pose(*ARM_POSE_DOOR_HANDLE)
        time.sleep(2)
        s.gripper.close()
        time.sleep(5)
        
        
        # 逐渐打开柜门。在每次循环中，机械臂以不同的角度移动，模拟打开柜门的动作。
        logger.info("Now i will use 7 steps to open the cab")   
        for i in range(7):
            # 计算当前迭代的角度，单位为弧度。这个角度用于后续计算机械臂的新位置和朝向。每次循环逐步增加打开柜门的角度
            d = 0.1 * (i + 1)  
            # 计算机械臂新的位置。基本思路是从初始位置（即抓住柜门把手的位置）开始，根据当前的角度d来调整位置。
            # [-0.4*math.sin(d), 0.4-0.4*math.cos(d), 0]计算了基于角度d的位置偏移量。这个偏移量模拟了当门打开时，门把手沿着一个半径为0.4的圆弧移动的路径。
            new_pos = ARM_POSE_DOOR_HANDLE[0] + np.array([-0.4*math.sin(d), 0.4-0.4*math.cos(d), 0])
            # 根据当前角度d计算新的朝向（四元数格式）。Rotation.from_euler方法用于根据欧拉角创建一个旋转对象。
            # 这里，旋转仅应用于z轴（[0, 0, -d]），表示柜门围绕z轴旋转打开。
            # 四元数是一种用于表示空间旋转的数学工具，它可以避免万向锁问题，并且计算效率较高。
            r = Rotation.from_euler("xyz", np.array([0, 0, -d]), degrees=False)
            new_ori = r.as_quat()
            # 命令机械臂移动到新的位置和朝向，模拟打开柜门的动作。
            s.arm.move_end_to_pose(new_pos, np.array(new_ori))
            time.sleep(0.5)
        
        # 打开夹爪
        logger.info("Suppose the cab is opened a little and i will loose the gripper")   
        s.gripper.open()
        time.sleep(5)

        # 这里每一步的位置和姿态参数都是根据具体任务需求、机器人的工作环境以及目标物体的位置精心计算和调试得出的，抓住已经打开了一半的门
        logger.info("I will move the arm and base to largely open the cab.")   
        s.arm.move_end_to_pose(np.array([0.3225, 0.00, 0.219]), np.array([0.0, 0.0, 0.0, 1.0]))
        s.arm.move_end_to_pose(np.array([0.3225, -0.25, 0.219]), np.array([0.0, 0.0, 0.0, 1.0]))
        s.arm.move_end_to_pose(np.array([0.5615004168820418, -0.2, 0.35123932220414126]), np.array([0.0, 0.0, 0.2953746452532359, 0.9547541169761965]))
        s.arm.move_end_to_pose(np.array([0.6015004168820418, -0.15, 0.35123932220414126]), np.array([0.0, 0.0, 0.2953746452532359, 0.9547541169761965]))
        
        # 机器人底座向后移动0.05米的操作，这可能是为了在操作后调整机器人的位置，向后把门大幅拉出
        # 然后机械臂移动回一个“标准移动”位置，这可能是一个安全位置或者准备进行下一步操作的位置
        s.arm.move_end_to_pose(np.array([0.4882092425581316, 0.2917225555849343, 0.3515424067641672]), np.array([0.0, 0.0, 0.6045684271573144, 0.7957869908463996]))
        back_pose = (np.array([-0.05, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        s.base.move_to(*back_pose, 'robot', True)
        s.arm.move_end_to_pose(*s.ARM_POSE_STANDARD_MOVING)
    
    logger.info("Suppose the cab is opened definitely and i will prepare for the next plan (place the pink/brown bowls)") 

    # 这段代码通过在不同的位置寻找目标物体（碗），并根据颜色将它们分类放置到不同的柜子中，展示了机器人在识别和操控物体方面的能力。
    # 通过调整观察位置、使用颜色作为分类依据，以及灵活地处理未找到目标物体的情况，这个过程展示了一种基本的自动化任务处理流程。
    logger.info("Now suppose the microware is OK. I plan to find bowls")
    obj_rgb = []
    combined_labels_for_bowl = [
        'bowl', 'cup', 'vessel', 'container', 'dish', 'basin', 'receptacle', 'pot', 
        'ceramicware', 'tableware', 'serveware', 'kitchenware', 'dinnerware', 
        'dishware', 'crockery', 'mixing bowl', 'salad bowl', 'Top-Down View of Bowl', 
        'Circular Rimmed Dish', 'Round Tableware Item', 'Open Cylindrical Vessel', 
        'Flat-Bottomed Serving Ware', 'Circular Kitchenware', 'Round Ceramic Piece', 
        'Dining Bowl Overhead', 'Open-top Porcelain Bowl', 'Gaming Bowl', 
        'Ceramic Bowl', 'Electronic Device Beside Dishware', 
        'Open-top Bowl Near Gaming Accessory', 'Entertainment-Themed Tableware', 
        'Modern Kitchenware with Tech', 'Interactive Gaming Bowl', 
        'Tech-Adjacent Porcelainware', 'Digital Lifestyle Ceramic', 
        'Game Controller and Bowl Set', 'Tech-Influenced Eating Utensil', 
        'Smart Home Dining Ware', 'Gamers Kitchen Bowl', 'Porcelain Bowl on Techy Surface', 
        'Augmented Dining Bowl'
    ]
    s.detector.set_classes(combined_labels_for_bowl)
    for j in range(5):
        s._prompt = 'bowl'
        cp = None
        s.base.move_to(*s.GRASP_POSE_2, 'world', False)
        look_num = []
        while cp is None:
            for direction in [1, 2]:
                if direction == 1:
                    s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_1)
                else:
                    s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_2)
                cp = s.lookforonce(0.4, 0.6)
                if cp is not None:
                    break
            look_num.append(1)
            if len(look_num)>2:
                break
        if len(look_num)>2:
            break
        centerpoint, object_mean_rgb = cp    
        centerp_car = np.linalg.inv(np.array(Rotation.from_quat(s.base.rotation).as_matrix())).dot((centerpoint-s.base.position))
        OBSERVE_ARM_POSE_TOP = (np.array([
                    centerp_car[0]- 0.2975 - 0.05,
                    centerp_car[1] + 0.17309,
                    0.018713334665877806,
                ]), np.array([
                    -0.13970062182177911,
                    0.6487791800204252,
                    0.032918235938941776,
                    0.7473190092439113,
                ]))
        s.arm.move_end_to_pose(*OBSERVE_ARM_POSE_TOP)
        time.sleep(1)
        s.grasp()

        # 通过计算得到碗的RGB颜色值，存储在obj_rgb列表中。这个颜色值用于后续判断碗应该放置的位置。
        # 对于找到的第一个碗（j == 0），直接放置到下层柜子中。
        # 对于之后找到的碗，比较它的颜色与第一个碗的颜色差异（通过abs(sum(obj_rgb[j]-obj_rgb[0]))计算）。
        # 如果颜色差异大于30，认为是不同类型的碗，应放置到上层柜子中；否则，放置到下层柜子中。
        obj_rgb.append(object_mean_rgb)
        if j != 0:
            logger.info(f"color: {abs(sum(obj_rgb[j]-obj_rgb[0]))}")
        if j == 0:
            s.place_bowl_lower()
        elif abs(sum(obj_rgb[j]-obj_rgb[0]))>30:
            s.place_bowl_upper()
        else:
            s.place_bowl_lower()

    # 第二个循环与第一个循环类似，不同之处在于寻找碗的起始位置（由GRASP_POSE_2定义）。这可能意味着机器人将从不同的位置或角度寻找碗，以确保能够找到更多碗。
    # 在第二个循环中，找到的碗的颜色（存储在obj_rgb_2列表中）会与第一个循环中找到的第一个碗的颜色进行比较，以决定放置的位置。如果第一个循环中没有找到任何碗（obj_rgb列表为空），则使用obj_rgb_2列表中的第一个碗的颜色作为参考。
    logger.info("I plan to change the base and continue to find bowls")
    obj_rgb_2 = []
    for j in range(5):
        s._prompt = 'bowl'
        cp = None
        s.base.move_to(*s.GRASP_POSE_1, 'world', False)
        look_num = []
        while cp is None:
            for direction in [1, 2]:
                if direction == 1:
                    s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_1)
                else:
                    s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_2)
                cp = s.lookforonce(0.4, 0.6)
                if cp is not None:
                    break
                look_num.append(1)
            if len(look_num)>2:
                break
        if len(look_num)>2:
            break
        centerpoint, object_mean_rgb = cp    
        centerp_car = np.linalg.inv(np.array(Rotation.from_quat(s.base.rotation).as_matrix())).dot((centerpoint-s.base.position))
        OBSERVE_ARM_POSE_TOP = (np.array([
                    centerp_car[0]- 0.2975 - 0.05,
                    centerp_car[1] + 0.17309,
                    0.018713334665877806,
                ]), np.array([
                    -0.13970062182177911,
                    0.6487791800204252,
                    0.032918235938941776,
                    0.7473190092439113,
                ]))
        s.arm.move_end_to_pose(*OBSERVE_ARM_POSE_TOP)
        time.sleep(1)
        s.grasp()
        obj_rgb_2.append(object_mean_rgb)
        if len(obj_rgb) != 0:
            if abs(sum(obj_rgb_2[j]-obj_rgb[0]))>30:
                s.place_bowl_upper()
            else:
                s.place_bowl_lower()
        else:
            if j == 0:
                s.place_bowl_lower()
            elif abs(sum(obj_rgb_2[j]-obj_rgb_2[0]))>30:
                s.place_bowl_upper()
            else:
                s.place_bowl_lower()
    
    # -----------------------------------------------------------------
    # 结束任务：最后，机器人的底座移动到结束位置，标志着任务的完成。
    logger.info("finish the task and i will return to the start position")
    s.base.move_to(*s.GRASP_POSE_1, 'world', False)
    s.base.move_to(*s.END_POSE, 'world', False) 
    time.sleep(120)
