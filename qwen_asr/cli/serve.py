# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

for _fn, _args in (
    (AutoConfig.register, ("qwen3_asr", Qwen3ASRConfig)),
    (AutoModel.register, (Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)),
    (AutoProcessor.register, (Qwen3ASRConfig, Qwen3ASRProcessor)),
):
    try:
        try:
            _fn(*_args, exist_ok=True)
        except TypeError:
            _fn(*_args)
    except Exception:
        pass

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration as _VllmASR
    from vllm import ModelRegistry
    try:
        ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", _VllmASR)
    except Exception:
        _alt = getattr(ModelRegistry, "register_and_maybe_inspect", None) or getattr(
            ModelRegistry, "register", None
        )
        if callable(_alt):
            _alt("Qwen3ASRForConditionalGeneration", _VllmASR)
        else:
            raise
except Exception as e:
    raise ImportError(
        "vLLM is not available, to use qwen-asr-serve, please install with: pip install qwen-asr[vllm]"
    ) from e

try:
    from vllm.entrypoints.cli.main import main as vllm_main
except ImportError:
    try:
        from vllm.entrypoints.cli.cli import main as vllm_main  # vLLM 0.30 新路径
    except ImportError:
        vllm_main = None  # type: ignore[assignment]

def main():
    if vllm_main is None:
        # 0.30 CLI 入口搬家后的最后回退：直接调 vllm serve 子进程
        import subprocess

        argv = [sys.executable, "-m", "vllm", "serve", *sys.argv[1:]]
        # qwen-asr-serve 的调用约定是 qwen-asr-serve <model> [vllm args]，
        # 这里已经 insert 过 serve，直接透传即可
        raise SystemExit(subprocess.call(argv))
    sys.argv.insert(1, "serve")
    vllm_main()


if __name__ == "__main__":
    main()