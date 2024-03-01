from concurrent import futures
import os

import grpc
import yaml

from .. import init_lm
from ..lm import *
from .service_pb2 import ExtractResponse
from .service_pb2_grpc import add_InstructionServiceServicer_to_server
from .service_pb2_grpc import InstructionServiceServicer


class InstructionService(InstructionServiceServicer):

    def __init__(self, local_cfg=None, base_model='chatglm') -> None:
        super().__init__()
        local_cfg = local_cfg or os.environ.get('LM_CONFIG', None)
        assert local_cfg is not None, "Config file path not found in argument or LM_CONFIG env var!"
        with open(local_cfg, 'r') as f:
            self.local_config = yaml.load(f, Loader=yaml.FullLoader)
        self.load_model(base_model)

    def load_model(self, model_name):
        torch.cuda.empty_cache()
        if hasattr(self, 'model_name'):
            if self.model_name == model_name:
                return
        if hasattr(self, 'model'):
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        self.model = init_lm(model_name, **self.local_config[model_name])
        self.model_name = model_name
        print(f"{model_name} loaded!")

    def extract(self, request, context) -> tuple:
        self.load_model(request.model)

        return ExtractResponse(status="success", responses=[self.model.infer(prompt=request.message)])


def init_lm_server(args=None, pipe=None, proxy=None, port=50051):
    print("proxy: ", proxy)
    if pipe is not None:
        pipe.send('init started')
        import os
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = os.environ['all_proxy'] = proxy
        os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['all_proxy'] = proxy
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InstructionServiceServicer_to_server(
        InstructionService(base_model=args.base_model if args is not None else 'gpt-3.5'), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    if pipe is not None:
        import time
        time.sleep(1)
        pipe.send('ready')
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='chatglm')
    args = parser.parse_args()
    init_lm_server(args)
