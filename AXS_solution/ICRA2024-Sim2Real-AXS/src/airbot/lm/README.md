# `airbot.lm`: Invoking Large Models

# Info

Supported models:

* `chatglm`: ChatGLM3-6B
* `gpt-3.5`: Proxy for OpenAI `gpt-3.5-turbo`
* `gpt-4`: proxy for OpenAI `gpt-4-1106-preview`


# How-to

## Invoke by RPC

A configuration file specifying remote url should exist. An example:

```yaml
server: www.tianbeiwen.com:50051
```

1. Invoke in CLI environment:
    ```shell
    LM_CONFIG=/path/to/config.yaml python -m airbot.lm.rpc --model-name [MODEL_NAME] --prompt [PROMPT]
    ``` 

2. Use as a module:

    Make sure the environment variable `LM_CONFIG` is set to the path to `config.yaml` 
    ```python
    import os
    os.environ['LM_CONFIG'] = '/path/to/config.yaml'

    from airbot.lm.rpc import query
    model_name = 'chatglm'
    prompt = 'hi'
    response = query(model_name, None, prompt)
    ```

## Invoke locally

Two environment variables should exist:

1. `CKPT_DIR`: the root directory for model checkpoints
2. `LM_CONFIG`: a yaml configuration file specifing the relative paths of model checkpoints. An example:

    ```yaml
    chatglm:
        path: chatglm3-6b

    baichuan-chat:
        path: Baichuan2-7B-Chat

    gpt-3.5:
        api_key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    gpt-4:
        api_key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    codegeex:
        path: codegeex2-6b

    deepseek-coder-1.3b-chat:
        path: deepseek-coder-1.3b-instruct

    deepseek-coder-6.7b-chat:
        path: deepseek-coder-6.7b-instruct

    deepseek-coder-33b-chat:
        path: deepseek-coder-33b-instruct

    codeshell-chat:
        path: CodeShell-7B-Chat

    codegen:
        path: codegen25-7b-multi

    phind-code-llama:
        path: Phind-CodeLlama-34B-v2

    ```

Then, to:

1. Invoke in CLI environment:

    ```shell
    LM_CONFIG=./local.yaml CKPT_DIR=/home/ubuntu/workspace/ckpts/ python -m airbot.lm --model-name [MODEL_NAME] --prompt [PROMPT]
    ```

2. Use as a module:

    ```python
    import os
    os.environ['LM_CONFIG'] = '/path/to/config.yaml'
    os.environ['CKPT_DIR'] = '/root/of/ckpts/'

    from airbot.lm import lm, query
    model_name = 'chatglm'
    prompt = 'hi'
    response = query(model_name, None, prompt)
    ```

## Deploy Inference Server

1. By Docker Compose:

    ```shell
    LM_CONFIG=$(pwd)/local.yaml CKPT_DIR=/home/ubuntu/workspace/ckpts docker compose up -d [--build]
    ```

2. Directly:

    `airbot` package should be installed first. See [here](../../README.md).
    ```shell
    LM_CONFIG=$(pwd)/local.yaml CKPT_DIR=/home/ubuntu/workspace/ckpts/ python -m airbot.lm.rpc.server
    ```

