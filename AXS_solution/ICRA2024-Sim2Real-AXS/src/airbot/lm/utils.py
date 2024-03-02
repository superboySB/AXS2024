import os
from typing import Dict, Optional, Union

import numpy as np
from torch.nn import Module


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM3
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike],
                       num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None,
                       **kwargs) -> Module:
    from transformers import AutoModel
    from transformers import AutoTokenizer
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


def mask_iou(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    assert mask_1.shape == mask_2.shape
    assert mask_1.dtype == mask_2.dtype == bool
    intersection = np.logical_and(mask_1, mask_2).sum()
    union = np.logical_or(mask_1, mask_2).sum()
    return intersection / union


def bbox_iou(box_1: np.ndarray, box_2: np.ndarray) -> float:
    # Convert from center coordinates to corner coordinates
    box1_x1 = box_1[0] - box_1[2] / 2
    box1_y1 = box_1[1] - box_1[3] / 2
    box1_x2 = box_1[0] + box_1[2] / 2
    box1_y2 = box_1[1] + box_1[3] / 2

    box2_x1 = box_2[0] - box_2[2] / 2
    box2_y1 = box_2[1] - box_2[3] / 2
    box2_x2 = box_2[0] + box_2[2] / 2
    box2_y2 = box_2[1] + box_2[3] / 2

    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area
    return iou

# depth2cloud函数的目的是将深度图像转换为点云。深度图像是一个二维数组，其中每个元素的值代表相机到对应像素点的距离。
# 点云是一系列在三维空间中的点，表示场景的几何形状。
# 这个转换过程需要考虑相机的内参矩阵（intrinsic_mat），它包含了相机的焦距（fx, fy）和光心坐标（cx, cy）等信息。
# 输入参数
# depth_im：深度图像，一个二维数组，其中的值表示从相机到物体表面的距离。
# intrinsic_mat：相机的内参矩阵，一个2x3或3x3的矩阵，包含相机的焦距和光心坐标。
# organized：一个布尔值，指示输出的点云是否应该保持与输入图像相同的组织结构（即是否是一个二维数组，每个像素对应一个三维空间点）。
# 输出
# cloud：生成的点云，根据organized参数，可以是一个三维数组（H x W x 3，每个像素对应一个三维坐标）或一个二维数组（H*W x 3，每行是一个三维坐标）。
def depth2cloud(depth_im, intrinsic_mat, organized=True):
    """ Generate point cloud using depth image only.
        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera_info: dict

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    # 获取深度图像尺寸：首先，获取深度图像的高度和宽度。
    height, width = depth_im.shape

    # 提取相机内参：从内参矩阵中提取焦距（fx, fy）和光心坐标（cx, cy）。
    fx, fy, cx, cy = intrinsic_mat[0][0], intrinsic_mat[1][1], intrinsic_mat[0][2], intrinsic_mat[1][2]
    assert (depth_im.shape[0] == height and depth_im.shape[1] == width)

    # 生成像素坐标网格：使用np.meshgrid函数，为深度图像的每个像素生成一个对应的网格坐标（xmap, ymap）。这个网格覆盖了整个图像平面。
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)

    # 计算三维坐标：
    points_z = depth_im  # change the unit to metel 直接从深度图像获取，代表每个点的深度值（Z坐标）。
    points_x = (xmap - cx) * points_z / fx # 通过将像素坐标转换为相机坐标系下的坐标来计算。这个转换考虑了相机的焦距和光心位置，
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud
