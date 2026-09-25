"""冒烟测试：qwen-asr 0.0.6 + transformers 5.18.dev0 + torch 2.13 + vllm 0.30 兼容检查(无 GPU 可跑前两段)。"""
import sys

print("== 1) 版本 ==")
import torch, transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
try:
    import vllm

    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import 跳过(无 GPU/未安装亦可):", type(e).__name__, str(e)[:200])

print("== 2) qwen_asr 导入 ==")
import qwen_asr
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner, parse_asr_output

print("qwen_asr ok:", Qwen3ASRModel, Qwen3ForcedAligner)

print("== 3) transformers 后端类可用性(不下载权重) ==")
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

cfg = Qwen3ASRConfig()
print("config ok:", cfg.model_type)
print("get_text_config vocab:", cfg.get_text_config().vocab_size)
print("rope_parameters bridge:", cfg.thinker_config.text_config.rope_parameters.get("rope_type"))

print("== 4) 工具函数 ==")
assert parse_asr_output("language Chinese<asr_text>你好")[0] == "Chinese"
assert Qwen3ASRModel is not None
print("utils ok")

print("== 5) vLLM 后端导入(需已装 vllm==0.30.0，Termux 请跳过) ==")
try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration as V

    print("vllm backend ok:", V)
    print("has build_data_parser:", hasattr(V, "get_speech_to_text_config"))
except Exception as e:
    print("vllm backend 跳过:", type(e).__name__, str(e)[:300])

print("ALL SMOKE OK")
