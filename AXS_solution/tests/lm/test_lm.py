import multiprocessing as mp
import os

import pytest

from airbot.lm.rpc import query
from airbot.lm.rpc.lm_server import init_lm_server


class TestLMRPC:

    @classmethod
    def setup_class(cls):
        cls.parent_pipe, cls.child_pipe = mp.Pipe()
        cls.p_server = mp.Process(target=init_lm_server,
                                  args=(None, cls.child_pipe, os.environ.get('HTTP_PROXY', None), 44444))
        cls.p_server.start()
        start_count = 0
        start_limit = 30
        while (not cls.parent_pipe.poll(1) or
               (recved := cls.parent_pipe.recv()) != 'ready') and start_count < start_limit:
            print(f'waiting for server to start: {recved}')
            start_count += 1
        assert start_count < start_limit, 'server start timeout'
        print('server started')

    @classmethod
    def teardown_class(cls):
        cls.p_server.terminate()

    @pytest.mark.parametrize(
        "model",
        [
            "chatglm",
            "gpt-4",
            "gpt-3.5",
            "codegeex",
            # "baichuan-chat",
            # "deepseek-coder-1.3b-chat",
            # "deepseek-coder-6.7b-chat",
            # "deepseek-coder-33b-chat",
            "codeshell-chat",
            # "codeshell",
            # "codegen-2.5",
        ])
    def test_lm(self, model):
        response = query(model, None, 'hello')
        print(response)
        assert response is not None and isinstance(response, str)
