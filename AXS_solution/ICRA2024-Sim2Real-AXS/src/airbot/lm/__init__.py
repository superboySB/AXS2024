import os

import yaml

from .prompt import PROMPTS
from .vlm import Detector, Segmentor

LMS = {}

VLMS = {}


def register_lm(name: str):

    def _register_lm(cls):
        LMS[name] = cls
        return cls

    return _register_lm


def register_vlm(name: str):

    def _register_vlm(cls):
        VLMS[name] = cls
        return cls

    return _register_vlm


def init_lm(name: str, *args, **kwargs):
    return LMS[name](*args, **kwargs)


def init_vlm(name: str, *args, **kwargs):
    return VLMS[name](*args, **kwargs)


_model = None
_model_name = None


def query(model_name: str, type: str, prompt: str, config_path: str = None):
    global _model, _model_name
    if config_path is None:
        config_path = os.environ.get('LM_CONFIG', None)
    assert config_path is not None, "Config file path not found in argument or LM_CONFIG env var!"
    with open(config_path, 'r') as f:
        local_config = yaml.load(f, Loader=yaml.FullLoader)
    if _model_name is None or _model_name != model_name:
        del _model
        _model = init_lm(model_name, **local_config[model_name])
        _model_name = model_name
    context = PROMPTS.get(type, '')
    return _model.infer(prompt=context + prompt)
