# %%
import warnings

warnings.filterwarnings("ignore")
from airbot.backend import Arm, Camera, Base, Gripper
import os
import numpy as np
import copy
from airbot.backend.utils.utils import camera2base, armbase2world
from airbot.lm import Detector, Segmentor
from airbot.grasp.graspmodel import GraspPredictor
from PIL import Image
import time
import cv2
from airbot.example.utils.draw import draw_bbox, obb2poly
from airbot.example.utils.vis_depth import vis_image_and_depth
from scipy.spatial.transform import Rotation
from threading import Thread, Lock
import math

from airbot.lm.utils import depth2cloud

os.environ['LM_CONFIG'] = "/root/Workspace/AXS_baseline/ICRA2024-Sim2Real-AXS/local.yaml"
os.environ['CKPT_DIR'] = '/root/Workspace/AXS_baseline/ckpt'



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
        0.2565699,
        0.2,
        0.171663168,
    ]), np.array([
        -0.13970062182177911,
        0.6487791800204252,
        0.032918235938941776,
        0.7473190092439113,
    ]))
    
    OBSERVE_ARM_POSE_2 = (np.array([
        0.2565699,
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

        self.detector = Detector(model='grounding-dino')
        # self.detector = Detector(model='yolo-v7')
        self.segmentor = Segmentor(model='segment-anything')
        self.grasper = GraspPredictor(model='graspnet')

        self.image_lock = Lock()
        self.result_lock = Lock()
        self.prompt_lock = Lock()
        self.running = True
        self.prompt = 'sky'
        self.update_once()
        self.t_vis = Thread(target=self.vis, daemon=True)
        self.t_vis.start()
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

    def update_once(self):
        with self.image_lock, self.result_lock:
            self._image = copy.deepcopy(self.camera.get_rgb())
            self._depth = copy.deepcopy(self.camera.get_depth())
            self._det_result = self.detector.infer(self._image, self._prompt)
            self._bbox = self._det_result['bbox'].numpy().astype(int)
            self._sam_result = self.segmentor.infer(self._image, self._bbox[None, :2][:, [1, 0]])
            self._mask = self._sam_result['mask']

    def update(self):
        while self.running:
            self.update_once()
            time.sleep(0.005)

    def vis(self):
        try:
            # Infinite loop to display images
            while self.running:
                image_draw = self.image
                image_draw = image_draw * (self.mask[:, :, None].astype(np.uint8) * 0.75 + 0.25)
                image_draw = draw_bbox(image_draw, obb2poly(self.bbox[None, ...]).astype(int))
                image_draw = image_draw.astype(np.uint8)
                image_show = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)
                cv2.imshow('RGB', image_show)
                _flag = np.any(self.mask)
                if _flag == True:
                    cv2.putText(image_show,
                                f"det score: {self._det_result['score']}, sam score: {self._sam_result['score']}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('RGB', image_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Exiting due to user interruption.")
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

    def grasp(self):
        # find the center of self.mask
        # center_cam = np.array(np.nonzeros(self.mask)).mean(axis=1)
        # center_frame = self.base_cloud[center_cam]

        # self.base.move_to(*self.OPERATE_BASE_POSE_1)
        # self.arm.move_end_to_pose(*self.OPERATE_ARM_POSE)
        # time.sleep(3)
        # method = input('method')
        method = "2"
        with self.image_lock, self.result_lock:
            _depth = copy.deepcopy(self._depth)
            _image = copy.deepcopy(self._image)
            _bbox = copy.deepcopy(self._bbox)
            _mask = copy.deepcopy(self._mask)

        cloud = self.base_cloud(_image, _depth, self.camera.INTRINSIC, self.CAMERA_SHIFT, self.arm.end_pose)

        if method == "1":
            direction = cloud[_bbox[0] - _bbox[2] // 2, _bbox[1]][:3] - self.arm.end_pose[0]
            direction = direction / np.linalg.norm(direction)
            grasp_position = cloud[_bbox[0] - _bbox[2] // 2 + 9, _bbox[1]][:3] - 0.12 * direction
            grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, np.pi / 2], degrees=False).as_quat()
        elif method == "2":
            grasp_position = cloud[ _bbox[0], _bbox[1] - _bbox[3] // 2 + 8][:3]
            grasp_position[2] = -0.165
            grasp_rotation = Rotation.from_euler('xyz', [0, np.pi / 2, 0], degrees=False).as_quat()
        else:
            bbox_mask = self._bbox2mask(_image, _bbox)
            (grasp_position, grasp_rotation), _ = self.grasper.infer(cloud, bbox_mask)

            grasp_rotation = Rotation.from_euler("yz", [np.pi / 2, np.pi / 2], degrees=False).as_quat()

        self.arm.move_end_to_pose(grasp_position, grasp_rotation)
        time.sleep(2)
        self.gripper.close()
        time.sleep(4)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

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

    def place_microwave(self):
        self.base.move_to(*self.BEFORE_MW_BASE_POSE, 'world', False)
        self.arm.move_end_to_pose(*self.ARM_POSE_TO_MICROWAVE)
        input_pose = (np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*input_pose, 'robot', True)
        self.gripper.open()
        time.sleep(2)
        output_pose = (np.array([-0.3, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*output_pose, 'robot', True)

    def close_microwave(self):
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)
        self.base.move_to(*self.POSE_CLOSE_MICROWAVE, 'world', False)
        self.arm.move_end_to_pose(*self.ARM_POSE_CLOSE_MICROWAVE)
        self.arm.move_end_to_pose(*self.ARM_POSE_CLOSE_MICROWAVE_END)
        output_pose1 = (np.array([-0.25, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
        self.base.move_to(*output_pose1, 'robot', True)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)


    def place_bowl_lower(self):
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)
        self.base.move_to(*self.POSE_TO_BOWL, 'world', False)
        self.arm.move_end_to_pose(*self.ARM_POSE_TO_LOWER_CABINET)
        input_pose = (np.array([0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.base.move_to(*input_pose, 'robot', True)
        self.gripper.open()
        self.arm.move_end_to_pose(*self.ARM_POSE_PUT_LOWER)
        time.sleep(2)
        output_pose = (np.array([- 0.35, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.base.move_to(*output_pose, 'robot', True)
        self.arm.move_end_to_pose(*self.ARM_POSE_STANDARD_MOVING)

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

    def lookforonce(self, det_th, sam_th):
        with self.image_lock, self.result_lock:
            _rgb = copy.deepcopy(self.camera.get_rgb())
            _depth = copy.deepcopy(self.camera.get_depth())
            _det_result = copy.deepcopy(self._det_result)
            _sam_result = copy.deepcopy(self._sam_result)
        _bbox = _det_result['bbox'].numpy().astype(int)
        _mask = _sam_result['mask']
        if np.any(_mask) is False:
            print(f"direction {direction} not Found")
            
        print("det score:", _det_result['score'])
        print("sam score:", _sam_result['score'])
        if _det_result['score'] > det_th and _sam_result['score'] > sam_th:
            print(f"Found the {self._prompt}")
            centerpoint = depth2cloud(_depth, self.camera.INTRINSIC, organized=True)[_bbox[0] // 1, _bbox[1] // 1]
            centerpoint = camera2base(centerpoint, self.CAMERA_SHIFT, self.arm.end_pose)
            centerpoint = (armbase2world(centerpoint, (self.base.position, self.base.rotation)).squeeze())
            object_rgb = _rgb[_bbox[0] - np.int32(_bbox[2]/4):_bbox[0] + np.int32(_bbox[2]/4), _bbox[1] - np.int32(_bbox[3]/4):_bbox[1] + np.int32(_bbox[3]/4)]
            mean_rgb = (np.mean(np.mean(object_rgb, axis=0), axis=0).astype(int))
            print('-' * 50)
            print('centerpoint is', centerpoint)
            print('object rgb is', mean_rgb)
            print('-' * 50)

            return centerpoint, mean_rgb

if __name__ == '__main__':
    s = Solution()

    s.base.move_to(*s.POSE_OPEN_CAB, 'world', False)
    time.sleep(1)
    POS_DOOR_HANDLE = np.array([0.30353946626186371, 1.230472918510437, 0])
    centerp_car = np.linalg.inv(np.array(Rotation.from_quat(s.base.rotation).as_matrix())).dot((POS_DOOR_HANDLE-s.base.position))
    ARM_POSE_DOOR_HANDLE = (np.array([
                    centerp_car[0] - 0.2975 - 0.01,
                    centerp_car[1] + 0.17309 - 0.01,
                    0.2,
                ]), np.array([
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]))
    s.arm.move_end_to_pose(*ARM_POSE_DOOR_HANDLE)
    s.gripper.close()
    time.sleep(5)
    
    for i in range(7):
        d = 0.1 * (i + 1)
        new_ori = np.array([0, 0, 0, 1])
        new_pos = ARM_POSE_DOOR_HANDLE[0] + np.array([-0.4*math.sin(d), 0.4-0.4*math.cos(d), 0])
        r = Rotation.from_euler("xyz", np.array([0, 0, -d]), degrees=False)
        new_ori = r.as_quat()
        s.arm.move_end_to_pose(new_pos, np.array(new_ori))
        time.sleep(0.5)
    
    s.gripper.open()
    time.sleep(3)
    s.arm.move_end_to_pose(np.array([0.3225, 0.00, 0.219]), np.array([0.0, 0.0, 0.0, 1.0]))
    s.arm.move_end_to_pose(np.array([0.3225, -0.25, 0.219]), np.array([0.0, 0.0, 0.0, 1.0]))
    s.arm.move_end_to_pose(np.array([0.5615004168820418, -0.2, 0.35123932220414126]), np.array([0.0, 0.0, 0.2953746452532359, 0.9547541169761965]))
    s.arm.move_end_to_pose(np.array([0.6015004168820418, -0.15, 0.35123932220414126]), np.array([0.0, 0.0, 0.2953746452532359, 0.9547541169761965]))

    s.arm.move_end_to_pose(np.array([0.4882092425581316, 0.2917225555849343, 0.3515424067641672]), np.array([0.0, 0.0, 0.6045684271573144, 0.7957869908463996]))
    back_pose = (np.array([-0.05, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))  
    s.base.move_to(*back_pose, 'robot', True)
    s.arm.move_end_to_pose(*s.ARM_POSE_STANDARD_MOVING)

    s._prompt = 'white mug'
    cp = None
    s.base.move_to(*s.GRASP_POSE_1, 'world', False)
    look_num = 0
    while cp is None:
        for direction in [1, 2]:
            if direction == 1:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_1)
            else:
                s.arm.move_end_to_pose(*s.OBSERVE_ARM_POSE_2)
            cp = s.lookforonce(0.65,0.65)
            if cp is not None:
                break
        look_num += 1
        if look_num>3:
            break
    if cp is not None:
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
        s.place_microwave()
    s.close_microwave()


    obj_rgb = []
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
                    cp = s.lookforonce(0.6, 0.6)
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
            obj_rgb.append(object_mean_rgb)
            if j != 0:
                print("color", abs(sum(obj_rgb[j]-obj_rgb[0])))
            if j == 0:
                s.place_bowl_lower()
            elif abs(sum(obj_rgb[j]-obj_rgb[0]))>30:
                s.place_bowl_upper()
            else:
                s.place_bowl_lower()
    obj_rgb_2 = []
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
                cp = s.lookforonce(0.6, 0.6)
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
    print("finish the task")
    s.base.move_to(*s.END_POSE, 'world', False) 
    time.sleep(20)
