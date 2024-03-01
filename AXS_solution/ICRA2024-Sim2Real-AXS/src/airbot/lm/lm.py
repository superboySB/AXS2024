import os
from typing import Optional

import torch
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from . import register_lm
from .utils import load_model_on_gpus


class LanguageModel:
    MODEL_CLASS = None

    def __init__(self, path, **kwargs) -> None:
        ckpt_base_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')
        path = os.path.join(ckpt_base_dir, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = self.MODEL_CLASS.from_pretrained(path, trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16).to('cuda:0')
        self.model = self.model.eval()

    def infer(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class ChatLanguageModel(LanguageModel):

    def __init__(self, path, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self.history = []

    def infer(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


@register_lm('chatglm')
class ChatGLM3(ChatLanguageModel):
    MODEL_CLASS = AutoModel

    def infer(self, prompt: str) -> str:
        response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history)
        print(self.history)
        return response


@register_lm('gpt-3.5')
class GPT3_5(ChatLanguageModel):
    VERSION = 'gpt-3.5-turbo'

    def __init__(self, api_key, **kwargs) -> None:
        self.history = []
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def infer(self,
              prompt: str,
              temperature: float = 0.1,
              stop_tokens: Optional[list[str]] = None,
              user_prompt: Optional[str] = None,
              **kwargs) -> str:
        is_first = len(self.history) == 0
        if is_first:
            self.history.append({"role": "system", "content": prompt})

        if user_prompt is not None:
            self.history.append({"role": "user", "content": user_prompt})
        elif prompt is not None:
            if not is_first:
                self.history.append({"role": "user", "content": prompt})
        else:
            raise RuntimeError("prompt or user_prompt must be provided!")
        print(self.history)
        response = self.client.chat.completions.create(
            messages=self.history,
            stop=stop_tokens,
            model=self.VERSION,
            temperature=temperature,
        ).choices[0].message
        self.history.append({"role": "assistant", "content": response.content.strip()})
        return response.content.strip()


@register_lm('gpt-4')
class GPT4(GPT3_5):
    VERSION = 'gpt-4-1106-preview'


@register_lm('baichuan-chat')
class BaichuanChat2(LanguageModel):
    MODEL_CLASS = AutoModelForCausalLM

    def __init__(self, path, **kwargs) -> None:
        from transformers.generation.utils import GenerationConfig
        super().__init__(path, **kwargs)
        self.model.generation_config = GenerationConfig.from_pretrained(path)

    def infer(self, prompt: str, **kwargs) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)
        return response


@register_lm('codegeex')
class CodeGeeX2(LanguageModel):
    MODEL_CLASS = AutoModel

    def infer(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_length=256, top_k=1)
        response = self.tokenizer.decode(outputs[0])
        return response


@register_lm('deepseek-coder-1.3b-chat')
@register_lm('deepseek-coder-6.7b-chat')
@register_lm('deepseek-coder-33b-chat')
class DeepSeekCoder(ChatLanguageModel):
    MODEL_CLASS = AutoModelForCausalLM

    def infer(self, prompt: str, **kwargs) -> str:
        self.history.append({'role': 'user', 'content': prompt})
        inputs = self.tokenizer.apply_chat_template(self.history, return_tensors="pt").to(self.model.device)
        # 32021 is the id of <|EOT|> token
        outputs = self.model.generate(inputs,
                                      max_new_tokens=512,
                                      do_sample=False,
                                      top_k=50,
                                      top_p=0.95,
                                      num_return_sequences=1,
                                      eos_token_id=32021)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        self.history.append({'role': 'assistant', 'content': response})
        return response


@register_lm('codeshell-chat')
class CodeShellChat(ChatLanguageModel):
    MODEL_CLASS = AutoModelForCausalLM

    def infer(self, prompt: str, **kwargs) -> str:
        response = self.model.chat(prompt, self.history, self.tokenizer)
        self.history.append((prompt, response))
        return response


@register_lm('codeshell')
class CodeShell(LanguageModel):

    def infer(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])


@register_lm('codegen-2.5')
class CodeGen(LanguageModel):

    def infer(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sample = self.model.generate(**inputs, max_length=128)
        response = self.tokenizer.decode(sample[0])
        return response
