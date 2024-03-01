from . import query
from .lm import *
from .vlm import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help="Text prompt")
    parser.add_argument('--model-name', type=str, help="Used model name", default='chatglm')
    parser.add_argument('--type', type=str, default='chat', help='Response type', choices=['chat'])
    parser.add_argument('--config-path', type=str, help="Path to Config", default=None)
    args = parser.parse_args()
    print(query(args.model_name, args.type, args.prompt, args.config_path))
