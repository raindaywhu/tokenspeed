# 代码地图与未闭合问题

本篇记录当前已经分析过的 TokenSpeed 源码范围、已经确认的结论和下一步还需要补的空白。它的作用是防止报告滑向“概念正确但证据不足”。

## 1. 已经分析过的代码区域

### 1.1 Runtime / Engine

核心入口：

- `python/tokenspeed/runtime/entrypoints/engine.py`
- `python/tokenspeed/runtime/engine/async_llm.py`
- `python/tokenspeed/runtime/engine/request_handler.py`
- `python/tokenspeed/runtime/engine/event_loop.py`
- `python/tokenspeed/runtime/engine/generation_output_processor.py`
- `python/tokenspeed/runtime/engine/output_processor.py`
- `python/tokenspeed/runtime/engine/data_parallel_controller.py`

已经确认：

- `Engine / AsyncLLM` 运行在主 Python 进程，负责请求接入、tokenize、front-end request state、collector、取消语义和 inline detokenize。
- `RequestHandler` 让 attention TP rank0 从 ZMQ 接请求，再广播给 TP group，保证每个 rank 的 C++ scheduler mirror 输入一致。
- `EventLoop` 不是简单胶水层，它负责 admission、cache result commit、`next_execution_plan()`、cache op submit、forward dispatch、output commit 和 scheduler advance。
- `event_loop_overlap()` 会先 dispatch 当前 forward，再 commit 上一轮 forward results，用于减少短 decode step 下 GPU forward 之间的 CPU gap。
- attention-DP 下，`DataParallelController` 可以按 queue 或 cache usage 分发请求；worker 侧还会做 DP metadata sync 和 idle forward。

### 1.2 C++ Scheduler / KV Ownership

核心入口：

- `tokenspeed-scheduler/bindings/python_module.cpp`
- `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp`
- `tokenspeed-scheduler/csrc/scheduler/operations/cache.cpp`
- `tokenspeed-scheduler/csrc/scheduler/outside_event_handler.cpp`
- `tokenspeed-scheduler/csrc/fsm/states.h`
- `tokenspeed-scheduler/csrc/fsm/forward_states.h`
- `tokenspeed-scheduler/csrc/fsm/forward_events.cpp`
- `tokenspeed-scheduler/csrc/fsm/cache_events.cpp`
- `tokenspeed-scheduler/csrc/resource/allocator/owned_pages.*`
- `tokenspeed-scheduler/csrc/resource/allocator/kv_allocator.*`
- `tokenspeed-scheduler/csrc/resource/page_container.*`
- `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp`
- `tokenspeed-scheduler/csrc/resource/radix_tree/tree_resource.h`

已经确认：

- `ExecutionPlan` 同时包含 `ForwardOp` 和 `CacheOp`，scheduler 不只决定算谁，也决定哪些 cache movement 要发生。
- Request FSM 状态对象持有资源：`DeviceNodeRef`、`HostNodeRef`、`LocalKVAllocator`、`ReqPoolIndex`、writeback pairs。
- `OwnedPages` 是 move-only RAII page owner，能减少 early-free、double-free、ownership 漏转移问题。
- prefix tree 接管 request-local full pages 是显式 ownership transfer，不是简单复制 page id。
- `scheduleDecode()` 会基于 tail page available tokens 和 reserve 计算是否需要新 page。
- `ScheduleRetractEvent` 是内存压力下的状态迁移：写回可恢复 KV，而不是简单丢弃。

### 1.3 GPU Forward 物化层

核心入口：

- `python/tokenspeed/runtime/execution/model_executor.py`
- `python/tokenspeed/runtime/execution/input_buffer.py`
- `python/tokenspeed/runtime/execution/forward_batch_info.py`
- `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py`
- `python/tokenspeed/runtime/layers/attention/registry.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/base.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`

已经确认：

- C++ Scheduler 管逻辑 page ownership，GPU `token_to_kv_pool` 管物理 KV tensor，中间由 `ModelExecutor` 通过 `req_to_page` 和 `InputBuffers` 物化。
- `update_block_table()` 把 `ForwardOp` 的 page list 写入 GPU block table。
- `reset_valid_cache_length()` 处理 prefill prefix length 和 retract recovery。
- fully prefix-cached prefill 可以是 zero-token model forward。
- `execute_idle_forward()` 让无 token rank 也参与分布式 collective，避免 DP/MoE 场景下通信挂住。

### 1.4 并行策略 / 通信

核心入口：

- `python/tokenspeed/runtime/utils/server_args.py`
- `python/tokenspeed/runtime/distributed/mapping.py`
- `python/tokenspeed/runtime/execution/distributed_initializer.py`
- `python/tokenspeed/runtime/distributed/comm_manager.py`
- `python/tokenspeed/runtime/distributed/comm_ops.py`
- `python/tokenspeed/runtime/models/base/placement.py`
- `python/tokenspeed/runtime/models/base/compiler.py`
- `python/tokenspeed/runtime/models/base/execution.py`
- `python/tokenspeed/runtime/models/base/decoder_layer.py`
- `python/tokenspeed/runtime/models/base/transformer_model.py`
- `python/tokenspeed/runtime/layers/moe/layer.py`
- `python/tokenspeed/runtime/layers/moe/dispatcher/deep_ep.py`
- `python/tokenspeed/runtime/models/deepseek_v4.py`

已经确认：

- CLI / server args 暴露 `--attn-tp-size`、`--dense-tp-size`、`--moe-tp-size`、`--data-parallel-size`、`--enable-expert-parallel` 等 split parallelism 控制。
- `Mapping` 明确分出 attention、dense、MoE 三套 layer family mapping。
- attention mapping 有 TP / CP / DP 槽位；dense mapping 有 TP / DP；MoE mapping 有 TP / EP / DP / TP_EP group。
- 当前源码禁止 MoE TP 和 EP 同时大于 1，所以不能声称 MoE 内部支持任意 TPxEP。
- `ENABLE_CP` 把 attention TP 参数解释成 CP size，说明 CP 还应谨慎降权。
- `CommManager` 按 `ForwardContext.global_num_tokens` / `input_num_tokens` 计算 token-aware scattered counts，并决定 all-reduce vs token all-gather / reduce-scatter。
- `CompiledDecoderLayer` / placement compiler 是通用基础设施；DeepSeek V4 当前实际路径更像手写模型 + `CommManager` + 自定义 attention/MoE。

### 1.5 DeepSeek V4 / latent-KV / MLA

核心入口：

- `python/tokenspeed/runtime/models/deepseek_v4.py`
- `python/tokenspeed/runtime/layers/attention/backends/deepseek_v4.py`
- `python/tokenspeed/runtime/layers/attention/deepseek_v4_ops.py`
- `python/tokenspeed/runtime/layers/attention/kv_cache/deepseek_v4.py`
- `tokenspeed-mla/`
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/`

已经确认：

- 当前 DeepSeek V4 文档示例仍使用 `--disable-kvstore`。
- `event_loop.py` 对 `DeepseekV4TokenToKVPool` + `enable_kvstore` 仍抛 `NotImplementedError`。
- `DeepseekV4TokenToKVPool.offload()` / `reload()` 明确未实现。
- DeepSeek V4 attention backend 有大量模型本地 metadata、paged cache table、compressed slot mapping、SWA/CSA decode/prefill 逻辑。
- vLLM 已支持 TokenSpeed 相关 MLA 算子路径，因此单个 MLA kernel 不应被作为最大护城河。

## 2. 已经确认的负面结论

这些结论需要在 PPT 中明确写出，避免过度包装：

1. **没有看到 tool-call-specific DDR KV offload。**  
   Tool call 相关处理目前主要在 chat template、stop token、normal finish path。没有看到专门的 `ToolCallEvent -> 将该 request KV 下沉 DDR` 策略。

2. **DeepSeek V4 hierarchical KVStore 当前不能作为短期收益。**  
   当前分支仍要求 DeepSeek V4 baseline 禁用 kvstore，V4 KV pool offload/reload 未实现。

3. **DeepSeek V4 不是已确认的 generic placement compiler 主路径。**  
   placement compiler 重要，但当前 V4 更依赖手写模型路径和 `CommManager`。

4. **MoE TP + EP 不能随意组合。**  
   源码有显式禁止 mixed TP/EP 的检查。报告应说 TokenSpeed 支持跨 layer family 的 split strategy，而不是 MoE 内部任意多维并行。

5. **MLA kernel 不是最大护城河。**  
   vLLM 已经吸收相关算子，剩余价值是端到端 latent-KV execution，而不是单 kernel。

## 3. 仍需要继续读的代码

### 3.1 DeepSeek V4 实际并行账本

还需要进一步拆：

- attention DP / dense TP / MoE EP 在 V4 推荐配置下的真实 rank group；
- hidden states 在 attention -> HC -> MoE -> residual 之间如何移动；
- `pre_mlp_comm()` / `post_mlp_comm()` 在 V4 各路径的真实 collective；
- MegaMoE / DeepEP / normal MoE backend 的 dispatch-combine 差异；
- shared expert dense TP 与 routed expert EP 是否产生额外 token movement。

目标输出应该是一张“V4 token movement ledger”：

```text
layer segment -> input placement -> collective -> output placement -> bytes/token -> latency counter
```

### 3.2 MemoryExecutor / HostExecutor overlap

还需要继续验证：

- cache op 是否真的能和 forward overlap；
- layerwise loadback 的 consumer index 如何与 attention layer 对齐；
- writeback stream fence 如何避免同一 device page 被复用后仍被旧 writeback 读取；
- storage backend / L3 query 在实际 workload 下的延迟是否会抵消收益。

### 3.3 EventLoop overlap 的 profiler 证据

代码能说明设计意图，但性能报告需要 profiler：

- forward-to-forward gap；
- CPU postprocess duration；
- scheduler.advance duration；
- grammar eager path 是否打断 overlap；
- output D2H copy event wait 是否成为新瓶颈。

### 3.4 Ascend / HCCL 适配风险

需要单独评估：

- NCCL 假设是否散落在 comm ops / backend；
- CUDA graph / stream / event 语义如何迁移到 Ascend；
- pinned CPU tensor / non_blocking copy 路径是否有等价实现；
- TokenSpeed kernel / MLA / V4 custom ops 哪些不可迁移，哪些只迁移 layout 和 metadata 思路。

### 3.5 vLLM-Ascend 可复制性对照

当前不需要继续深挖 vLLM 主实现，但 PPT 仍需要一个复制难度矩阵：

| TokenSpeed 机制 | vLLM/vLLM-Ascend 复制级别 | 需要判断的问题 |
|---|---|---|
| split parallelism CLI | config-level | 是否已有等价参数和 group 初始化 |
| model-specific parallel wiring | model-patch | 是否能在 V4/Kimi model runner 中局部复制 |
| token-aware comm | runtime-protocol | 是否已有跨 DP token metadata 和 idle forward |
| Scheduler/KV ownership FSM | architecture-level | 是否需要重构 scheduler/block manager 生命周期 |
| cache op / forward op combined plan | architecture-level | 是否能把 async cache movement 纳入 scheduler plan |
| placement compiler | architecture / model-runtime | 是否需要在模型图层表达 placement transition |
| MLA kernel | backend-level | vLLM 已有相关路径，复制难度下降 |

## 4. 下一轮研究产物建议

建议下一轮不要继续扩散到所有模块，而是集中产出三张能直接进 PPT 的图/表：

1. **TokenSpeed runtime 组件图**  
   必须同时画出 Main Python、DP Controller、Scheduler Worker、C++ Scheduler、GPU execution plane。

2. **一条请求的状态流图**  
   从 `GenerateReqInput` 到 `BatchTokenIDOut`，标注哪些状态在 Python、哪些在 C++、哪些在 GPU。

3. **DeepSeek V4 并行账本**  
   attention / dense / MoE 每段的 group、collective、token count、可能收益、复制难度。

这三张图完成后，PPT 的深度才会从“功能说明”变成“系统机制解释”。
