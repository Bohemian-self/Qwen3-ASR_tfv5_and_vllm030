# qwen-asr 0.0.6 + transformers 5.18.0.dev0 + torch 2.13.0 + vllm 0.30.0 兼容补丁说明

本次已直接修改 `qwenasr_vllm_optimizated/Qwen3-ASR` 源码(保持版本号 0.0.6)，使同一份代码在
旧版(transformers 4.57/vllm 0.17)与新版(transformers 5.18.dev0/vllm 0.30)都能 import。

## 2026-09-25 追补：vLLM 0.30 运行时 `Can't extract 'str' to 'Vec'` 修复

报错链：`Qwen3ASRModel.LLM -> vLLM init -> get_dummy_mm_inputs -> processor.apply
-> _maybe_apply_prompt_updates(vllm/model_executor/models/qwen3_omni_moe_thinker.py:1357)
-> _apply_prompt_updates_via_text -> _plan_prompt_updates_with -> _find_queue_match
-> _iter_text_matches -> tokenizer.decode(target) -> TypeError: Can't extract 'str' to 'Vec'`

根因：vLLM 0.30 的 `PromptUpdate.target` 类型为 `list[int] | PromptIndex`
(`vllm/multimodal/processing/processor.py`)，`_iter_text_matches` 内会对 target 做
`tokenizer.decode(target)`。旧 `qwen-asr` 传 `target=audio_token(str)`，
decode(str) 直接炸；父类 `Qwen2_5OmniThinkerMultiModalProcessor` 正确写法是
`target=[audio_token_id]`。

本次修复（`qwen_asr/core/vllm_backend/qwen3_asr.py`）：
- `_get_prompt_updates` 的 `PromptReplacement(target=audio_token)` → `target=[audio_token_id]`，
  `replacement` 保持返回 `[audio_token_id]*num_features` 不变；
- `audio_token_id` 解析加回退：`vocab[]` → `convert_tokens_to_ids` → `config.audio_token_id(151646)`；
- `get_hf_config()` 三级回退并在 `thinker_config is None` 时显式报错，
  替代旧的静默默认（旧日志 `thinker_config is None. Initializing ... default values` 会导致
  音频 token 数/采样率全错后再在 prompt 更新处连环炸）；
- `Qwen3ASRForConditionalGeneration.__init__` 对 `hf_config.thinker_config is None` 显式报错，
  提示先执行 `AutoConfig.register('qwen3_asr', Qwen3ASRConfig)` 且用原生权重 ID；
- `get_dummy_text/get_hf_processor` 的 `audio_token` 加 `getattr` 回退。

Kaggle/服务器注意：你报错栈里加载的是 `/usr/local/lib/python3.12/dist-packages/qwen_asr`
（PyPI 旧包），不是本仓库补丁。必须先卸载旧包再以 `--no-deps` 装本仓库，否则改了也不生效：
```bash
pip uninstall -y qwen-asr
pip install --no-deps -e /path/to/Qwen3-ASR
# 或 pip install --no-deps /path/to/Qwen3-ASR
```

## 改了哪些文件（初版）

1. `pyproject.toml`
   - `transformers~=4.57.6` -> `transformers>=4.57.6`，`torch>=2.4`，`vllm>=0.17`
   - 修 `package-data` typo：`qwen_tts` -> 同时保留 `qwen_asr`+`qwen_tts`

2. `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
   - 全部 transformers 导入加 try/except 回退
   - 新增 `_resolve_attention_interface/_get_rope_params/_make_causal_mask`
   - RoPE：兼容 `rope_parameters`(5.x)与`rope_scaling/rope_theta`(4.x)；`ROPE_INIT_FUNCTIONS["default"]`缺失回退
   - `mrope_section` 用 getattr 安全取
   - 3处 attention forward：去 `deprecate_kwarg(version=4.58)`，手动兼容 `past_key_value`；`Cache.update(sin/cos)` 双签名；`ALL_ATTENTION_FUNCTIONS[]` -> `_resolve_attention_interface`
   - `Qwen3ASRPreTrainedModel._skip_keys_device_placement: str` -> `["past_key_values"]`
   - `ThinkerTextModel.forward`：`@check_model_inputs()` -> try两种写法；`DynamicCache(config=top)`失败回退 `get_text_config()/空参`；`get_seq_length()` try；`create_causal_mask(input_embeds/cache_position)` -> `_make_causal_mask`
   - `loss_function(vocab_size=)` -> 依次试 `vocab_size/config/位置参数`
   - `Qwen3ASRForConditionalGeneration.generate` 新增 `generation_config/cache_implementation` 透传

3. `configuration_qwen3_asr.py`
   - `Qwen3ASRTextConfig.rope_parameters` 只读桥接属性
   - `ThinkerConfig.get_text_config` 显式返回 `text_config`
   - `Qwen3ASRConfig.get_text_config` 深钻到 `thinker.text_config`，保证 `DynamicCache/vLLM` 拿到真 text config

4. `processing_qwen3_asr.py`
   - `BatchFeature/AudioInput` 双路径导入；`valid_processor_kwargs` 声明
   - `__init__` 改关键字传参；`audio_token/bos/eos` getattr 回退
   - `__call__`：`attention_mask/feature_attention_mask`、`input_features/input_values` 双兼容；`return_tensors` 从 merged kwargs 取

5. `qwen_asr/core/vllm_backend/qwen3_asr.py`
   - 全量 vllm 导入桥接(0.17~0.30)：config/distributed/inputs/activation/attention/linear/weight_utils/interfaces/module_mapping/qwen3/omni_thinker/utils/whisper langs/multimodal/sequence/attention enum/tokenizer/processor/vision
   - `Qwen3OmniMoeThinker` 缺失回退 `BaseMultiModalProcessor`+`dict`
   - `_resolve_activation_fn`；`get_vit_attn_backend` try；`compute_attn_mask_seqlen` 枚举 getattr
   - `ProcessingInfo` 新增 `build_data_parser()` 别名(修 0.16+ `ValueError: ...moved to build_data_parser`)
   - `embed_input_ids` merge 函数缺失回退；`get_speech_to_text_config` 双签名；`supported_languages.get` 兼容 list/dict

6. `qwen_asr/inference/qwen3_asr.py`
   - `Auto*register(exist_ok=True)` 安全注册；`ModelRegistry` 双入口
   - 新增 `_filter_llm_kwargs/_make_sampling_params/_vllm_generate`
   - `LLM()`：5参默认值保留，`kwargs` 按签名过滤；`SamplingParams(max_tokens/max_new_tokens)` 双试；`Processor.from_pretrained(trust_remote_code=True)` try
   - `_infer_asr_transformers`：`device/dtype` 安全取；`BatchFeature.to` 分开转；`generate` 返回 tensor/对象双兼容
   - `_infer_asr_vllm/streaming/finish`：统一 `prompt_token_ids+audio(ndarray)`，去 `use_tqdm=False`(按签名)，`tokenizer.encode(add_special_tokens=False)` try

7. `qwen_asr/inference/qwen3_forced_aligner.py`
   - 同上安全注册；`AutoProcessor(trust_remote_code)` try；`inputs.to(device/dtype)` 分开转

8. `qwen_asr/cli/serve.py`
   - 同上安全注册；`vllm.entrypoints.cli.main/cli.cli` 双路径；`vllm_main is None` 回退子进程

## 安装(服务器 Linux+CUDA，不要在 Termux 跑 vLLM)

```bash
pip install -U "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu129
pip install -U --pre "transformers==5.18.0.dev0"
# 若 PyPI 无此 dev 版：pip install -U "git+https://github.com/huggingface/transformers@main"
pip install -U "vllm==0.30.0" "vllm[audio]==0.30.0"
pip install --no-deps -e /data/data/com.termux/files/home/qwenasr_vllm_optimizated/Qwen3-ASR
python -c "import transformers,vllm,torch,qwen_asr; print(transformers.__version__, vllm.__version__, torch.__version__)"
```

## 已知风险(社区一致结论)

- transformers 后端在 5.3/5.4 已观测精度下降(#138)；5.18.dev0 未验证，生产只建议 vLLM 后端。
- vLLM 0.30 `enable_chunked_prefill=False/max_model_len=16384` 仍透传，若 0.30 校验收紧会被 `_filter_llm_kwargs` 丢弃，以 vLLM 默认值为准。
- 长音频/大批量仍走 `max_inference_batch_size` 分片；`torch.compile` 在 MoE 模型上保持 `_can_compile_fullgraph=False`。
