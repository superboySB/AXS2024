from concurrent import futures
import io
import os

import grpc
import numpy as np
import torch
import yaml

from .. import init_vlm
from ..vlm import *
from .service_pb2 import BoundingBox
from .service_pb2 import DetectResponse
from .service_pb2 import Image
from .service_pb2 import Point
from .service_pb2 import SegmentResponse
from .service_pb2_grpc import add_ImageServiceServicer_to_server
from .service_pb2_grpc import ImageServiceServicer


class ImageService(ImageServiceServicer):

    def __init__(self, local_cfg=None, base_model=None):
        super().__init__()
        local_cfg = local_cfg or os.environ.get('LM_CONFIG', None)
        assert local_cfg is not None, "Config file path not found in argument or LM_CONFIG env var!"
        with open(local_cfg, 'r') as f:
            self.local_config = yaml.load(f, Loader=yaml.FullLoader)
        self.detection_model = None
        self.detection_model_name = None
        self.segmentation_model = None
        self.segmentation_model_name = None

    def load_model(self, model_name):
        model_type = self.local_config.get(model_name, {}).get('type', 'None')
        if model_type == 'detection':
            self.load_det_model(model_name)
        elif model_type == 'segmentation':
            self.load_seg_model(model_name)
        else:
            print('Model nonexists or type unspecified')

    def load_seg_model(self, model_name):
        if self.segmentation_model is not None and model_name != self.segmentation_model_name:
            del self.segmentation_model
            self.segmentation_model = None
            self.segmentation_model_name = None
            torch.cuda.empty_cache()
        self.segmentation_model_name = model_name
        self.segmentation_model = Segmentor(model_name)
        print(f"{model_name} loaded!")

    def load_det_model(self, model_name):
        if self.detection_model is not None and model_name != self.detection_model_name:
            del self.detection_model
            self.detection_model = None
            self.detection_model_name = None
            torch.cuda.empty_cache()
        self.detection_model_name = model_name
        self.detection_model = Detector(model_name)
        print(f"{model_name} loaded!")

    def segment(self, request, context):
        self.load_model(request.model)
        img_arr = np.frombuffer(
            request.image.data,
            dtype=np.uint8,
        ).reshape(request.image.height, request.image.width, -1)
        ret = self.segmentation_model.infer(image=img_arr, prompt=np.array([(request.prompt.x, request.prompt.y)]))
        return SegmentResponse(status="success",
                               mask=Image(
                                   data=ret['mask'].astype(np.uint8).tobytes(),
                                   height=ret['mask'].shape[0],
                                   width=ret['mask'].shape[1],
                               ),
                               score=ret['score'])

    def detect(self, request, context):
        self.load_model(request.model)
        img_arr = np.frombuffer(
            request.image.data,
            dtype=np.uint8,
        ).reshape(request.image.height, request.image.width, -1)
        ret = self.detection_model.infer(img_arr, request.target)
        box = ret['bbox']
        bbox = BoundingBox(
            center=Point(x=int(box[0].item()), y=int(box[1].item())),
            height_x=int(box[2].item()),
            width_y=int(box[3].item()),
            score=float(ret['score']),
        )
        return DetectResponse(status="success", boxes=[bbox])


def init_vlm_server(args=None, pipe=None, proxy=None, port=50051):
    print("proxy: ", proxy)
    if pipe is not None:
        pipe.send('init started')
        import os
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = os.environ['all_proxy'] = proxy
        os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['all_proxy'] = proxy
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ImageServiceServicer_to_server(
        ImageService(base_model=args.base_model if args is not None else 'segment-anything'),
        server,
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    if pipe is not None:
        import time
        time.sleep(1)
        pipe.send('ready')
    server.wait_for_termination()
