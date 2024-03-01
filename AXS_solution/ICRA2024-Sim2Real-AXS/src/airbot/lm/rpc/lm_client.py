import os

import grpc
import yaml

# from ..prompt import PROMPTS
from .service_pb2 import ExtractRequest
from .service_pb2_grpc import InstructionServiceStub
from .utils import parse_config


def query(model_name: str, type: str, prompt: str, config_path: str = None):
    local_config = parse_config(config_path)
    context = ""
    server = local_config['server']
    with grpc.insecure_channel(server) as channel:
        stub = InstructionServiceStub(channel)
        response = stub.extract(ExtractRequest(model=model_name, type=type, message=context + prompt))

    return response.responses[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help="Text prompt")
    parser.add_argument('--model-name', type=str, help="Used model name", default='chatglm')
    parser.add_argument('--type', type=str, default='chat', help='Response type', choices=['chat'])
    parser.add_argument('--config-path', type=str, help="Path to Config", default=None)
    args = parser.parse_args()
    print(query(args.model_name, args.type, args.prompt, args.config_path))
