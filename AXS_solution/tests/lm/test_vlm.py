import multiprocessing as mp
import os

from importlib_resources import files
import numpy as np
from PIL import Image
import pytest

from airbot.lm.rpc import det_query
from airbot.lm.rpc import seg_query
from airbot.lm.rpc.vlm_server import init_vlm_server
from airbot.lm.utils import bbox_iou
from airbot.lm.utils import mask_iou


class TestVLMRPC:

    @classmethod
    def setup_class(cls):
        cls.parent_pipe, cls.child_pipe = mp.Pipe()
        cls.p_server = mp.Process(target=init_vlm_server,
                                  args=(None, cls.child_pipe, os.environ.get('HTTP_PROXY', None), 44444))
        cls.p_server.start()
        start_count = 0
        start_limit = 60
        while (not cls.parent_pipe.poll(1) or
               (recved := cls.parent_pipe.recv()) != 'ready') and start_count < start_limit:
            print(f'waiting for server to start: {recved}')
            start_count += 1
        assert start_count < start_limit, 'server start timeout'
        print('server started')

    @classmethod
    def teardown_class(cls):
        cls.p_server.terminate()

    def test_segment_response(self):
        for _ in range(3):
            img_path = files('airbot.data').joinpath('test_apple.jpeg')
            img_arr = np.array(Image.open(img_path.as_posix()))
            img_center = (img_arr.shape[0] // 2, img_arr.shape[1] // 2)
            response = seg_query('segment-anything', img_arr, img_center)
            assert response.shape == (*img_arr.shape[:2], 1)
            assert response.dtype == bool
            assert response.max() == 1
            assert response.min() == 0

    def test_segment_iou(self):
        img_path = files('airbot.data').joinpath('test_apple.jpeg')
        img_arr = np.array(Image.open(img_path.as_posix()))
        img_center = (img_arr.shape[0] // 2, img_arr.shape[1] // 2)
        response = seg_query('segment-anything', img_arr, img_center)
        gt_mask_path = files('airbot.data').joinpath('test_apple_mask.npy')
        gt_mask_arr = np.load(gt_mask_path.as_posix())[:, :, None]
        assert mask_iou(response, gt_mask_arr) > 0.8

    def test_detect_response(self):
        for _ in range(3):
            img_path = files('airbot.data').joinpath('test_apple.jpeg')
            img_arr = np.array(Image.open(img_path.as_posix()))
            response = det_query('grounding-dino', img_arr, 'apple')
            assert isinstance(response, list)
            assert len(response) > 0
            assert isinstance(response[0], list)
            assert len(response[0]) == 4

    def test_detect_iou(self):
        img_path = files('airbot.data').joinpath('test_apple.jpeg')
        img_arr = np.array(Image.open(img_path.as_posix()))
        response = det_query('grounding-dino', img_arr, 'apple')
        ious = []
        gt_box = [192, 192, 278, 278]
        print(response)
        for bbox in response:
            ious.append(bbox_iou(bbox, gt_box))
        print(ious)
        assert max(ious) > 0.9
