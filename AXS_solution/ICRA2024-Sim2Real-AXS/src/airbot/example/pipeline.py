import os

import numpy as np
from airbot.backend import AirBot
from airbot.lm.vlm import Detector, Segmentor
from airbot.lm.utils import depth2cloud
from airbot.grasp.graspmodel import GraspPredictor

os.environ['CKPT_DIR'] = '/home/ubuntu/workspace'

if __name__ == "__main__":
    # initialize backend
    agent = AirBot(arm_backend="ros", base_backend="ros", camera_backend="ros", gripper_backend='ros')
    if state := agent.init():
        print("AirBot init success")
    # get camera information
    camera_info = {'height': agent.camera.HEIGHT, 'width': agent.camera.WIDTH, 'intrinsic': agent.camera.INTRINSIC}
    # initialize detect model
    detectmodel = Detector(model='groundingdino',
                           ckpt=os.path.join('gd', 'groundingdino_swinb_cogcoor.pth'),
                           config=os.path.join('gd', 'GroundingDINO_SwinB_cfg.py'))
    # initialize segment model
    segmentmodel = Segmentor(model='sam', ckpt=os.path.join('sam', 'sam_vit_l_0b3195.pth'), model_type='vit_l')
    #initialize grasp model
    graspmodel = GraspPredictor(model='graspnet',
                                ckpt=os.path.join('graspnet', 'checkpoint-rs.tar'),
                                camera_info=camera_info)
    # infer satge
    color = agent.camera.get_rgb()
    depth = agent.camera.get_depth()
    prompt = 'white bowl'
    # get prompt
    detect = detectmodel.infer(color, prompt=prompt)
    bbox = detect['bbox']
    # get mask
    segment = segmentmodel.infer(color, bbox)
    mask = segment['mask']
    # get pose
    camera_cloud = depth2cloud(depth, camera_info)
    shift = [-0.065, 0, 0.0524]  # -x_gripper2arm + x_camera2arm, y, z  ##simbot
    #shift = [-0.225 + 0.07475, 0, 0.067]  # -x_gripper2arm + x_camera2arm, y, z  ##realbot
    current_pose = agent.arm.end_pose()
    transed_cloud = agent.camera2base(camera_cloud, shift, current_pose)
    cloud = np.stack[transed_cloud, color].reshape([-1, 6])
    pose = graspmodel.infer(cloud, mask)
    # grasp stage
    agent.arm.move_end_to_pose(pose)
    agent.gripper.close()
