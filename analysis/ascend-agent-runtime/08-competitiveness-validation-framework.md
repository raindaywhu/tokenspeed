# TokenSpeed 竞争力验证框架

## 1. 目标

本文定义如何评估 TokenSpeed 在 agent workload 下，相比 vLLM 和 TensorRT-LLM 是否真的具备优势。

本文不预设 TokenSpeed 更好，而是把结论拆成几类：

- 已验证优势
- 潜在优势
- 架构差异
- 能力持平
- 劣势
- 需要 benchmark 或 profiling 证据的未知项

不能只因为 TokenSpeed 架构不同，就把某个能力描述成优势。只有当这种差异在相同 workload、模型、硬件和 serving 约束下带来可测量收益时，才应描述为优势。

## 2. 什么才算优势

一个 TokenSpeed 能力只有同时满足三个条件，才可以归类为已验证优势。

1. 架构差异

   TokenSpeed 有一个与 vLLM 和 TensorRT-LLM 显著不同的 runtime 机制。

2. 可观测收益

   该机制改善了某个可测量的 serving 属性，例如：

   - 更低的 CPU 控制面开销
   - 更小的 forward-to-forward gap
   - 更低的 TTFT
   - 更低的 TPOT
   - tool-call 工作流中更低的 Round 2 TTFT
   - 更高的 prefix cache hit rate
   - 更少的 KV eviction 或 recomputation
   - agent traffic 下更高的 GPU 利用率
   - mixed prefill/decode/tool workload 下更好的尾延迟

3. 可复现实验证据

   该收益在受控 benchmark 中可复现，并且至少控制这些变量：

   - 相同模型
   - 相同硬件
   - 相同量化方式
   - 相同请求分布
   - 相同 tool schema 大小
   - 相同并发水平
   - 可比的 tokenizer 和 chat template 行为
   - 可比的 routing 和 cache locality 策略

如果只满足条件 1，应称为潜在优势。

如果条件 1 和条件 2 在机制上合理，但尚未测量，应称为假设。

如果收益依赖的代码路径尚未实现，或目标模型路径未启用该能力，应称为未验证。

## 3. 结果标签

后续分析中应统一使用这些标签。

| 标签 | 含义 |
|---|---|
| 已验证优势 | 有源码、profiling 和可复现 benchmark 结果共同支撑 |
| 潜在优势 | 有架构分析支撑，但缺 benchmark 证据 |
| 能力持平 | vLLM 或 TensorRT-LLM 已有类似能力 |
| 劣势 | TokenSpeed 更弱、更不成熟或能力更不完整 |
| 未知 | 缺少足够源码证据或 benchmark 证据 |
| 不适用 | 该比较维度不适用于选定 runtime 或模型路径 |

## 4. 决策矩阵

| 维度 | TokenSpeed | vLLM | TensorRT-LLM | 当前判断 |
|---|---|---|---|---|
| OpenAI-compatible protocol 处理 | 通常由 SMG gateway 或 serving adapter 处理 | OpenAI-compatible serving path 成熟 | 通常由 Triton、自定义 server 或上层 adapter 处理 | 不是 TokenSpeed engine 优势 |
| Anthropic / Claude-style protocol 处理 | gateway 或 adapter 层责任 | 可通过 serving adapter 实现，不是 EngineCore 职责 | 通常是上层职责 | protocol 支持不是 engine-level 优势 |
| tool-call parser | gateway / parser 层 | API serving / ToolParser 层 | 通常是上层 adapter 或 server integration | 能力持平或集成差异，不是 engine 优势 |
| tool-call lifecycle | tool loop 大概率在 engine 外部 | tool parser 和 tool loop 在 EngineCore 外部处理 | 通常在 runtime backend 外部处理 | engine-level 优势未证明 |
| same-request tool wait and resume | 未证明 | 默认不是该行为 | 默认不是该行为 | 未知 / 不应声称 |
| tool-call-specific DDR KV offload | 未证明 | V1 默认不依赖 GPU-CPU KV swapping | 取决于 KV cache manager 和部署方式 | 未知 / 不应声称 |
| 多轮 prefix reuse | 可能通过 prefix cache、routing 和 KV lifecycle 实现 | prefix caching 和 KV manager 能力成熟 | 按 runtime 配置支持 KV reuse/offload | 需要 benchmark |
| scheduler control plane | C++ Scheduler 和 request FSM 可能降低开销 | EngineCore scheduler 成熟且广泛使用 | C++ runtime、inflight batching 和 GPU execution stack 强 | TokenSpeed 潜在优势，未验证 |
| agent workload 的 CPU 固定开销 | SMG + Python + C++ scheduler 混合路径 | API server + EngineCore + workers | Triton/server + C++ backend 路径 | 需要 profiling |
| forward-to-forward gap | 如果 scheduler 和 output path 足够高效，可能降低 | 需要测量 | graph/runtime 优化路径通常较强 | 需要 profiling |
| KV ownership model | TokenSpeed 强调 logical KV ownership 和 page lifecycle | vLLM 有成熟 KV block manager 和 prefix cache | TensorRT-LLM 有成熟 KV cache 机制 | 只有测量后才可称为潜在优势之外的结论 |
| DeepSeek / MLA specialization | 是 TokenSpeed 重要目标路径 | 取决于模型实现和 kernel 支持 | NVIDIA kernel/runtime 生态强 | 没有模型级 benchmark 前未知 |
| MoE 和通信成熟度 | 需要验证 TP/EP/DP/CP 组合 | distributed serving 生态成熟 | NCCL/NVIDIA stack 强 | TokenSpeed 可能较不成熟 |
| 生产成熟度 | preview / early-stage | 生产使用成熟 | 生产使用成熟 | TokenSpeed 劣势 |
| 可观测性和工具链 | 需要显式 counter 和 profiler 支持 | 生态和 metrics 成熟 | 部署/profiling stack 成熟 | 除非补齐，否则 TokenSpeed 劣势 |
| Ascend 可迁移性 | 若 CUDA/NCCL 假设被隔离，战略上可能可行 | 需要 backend 支持和适配 | NVIDIA-centric | TokenSpeed 潜在战略价值，但工程风险高 |
| kernel 成熟度 | 取决于模型路径和 custom kernels | backend 支持广 | NVIDIA kernel stack 很强 | TokenSpeed 不自动占优 |
| agent workload fit | 如果 scheduler/KV/runtime integration 被验证，可能较强 | 强 baseline，不能当弱基线 | backend 强，但上层 agent loop 可能外置 | 需要受控 benchmark |

## 5. 核心解释

TokenSpeed 可能的优势不是泛泛的 tool-call support。

Tool-call parsing、protocol conversion、MCP/tool execution loop 通常是 gateway 或 serving-layer 职责。vLLM 也把 tool-call parsing 放在 OpenAI-compatible serving path，而不是 EngineCore 或 GPU workers 里。

因此关键问题不是：

```text
TokenSpeed 是否支持 tool call？
```

关键问题是：

```text
TokenSpeed 的 scheduler、request FSM、KV lifecycle 和 execution plan
是否降低了多轮 agent serving 的成本，
并且相比 vLLM 和 TensorRT-LLM 有可测量收益？
```

这必须通过测量回答。

## 6. 竞争力 claim ledger

### Claim 1：TokenSpeed C++ Scheduler 可能降低 agent CPU 控制面开销

状态：潜在优势，未验证。

当前证据：

- TokenSpeed 有嵌入式 C++ Scheduler 和 request FSM。
- Scheduler 看起来持有 request lifecycle 和 logical KV page state。
- 如果关键路径确实从 Python 移入 C++，该设计可能降低 Python-side scheduling overhead。

缺失证据：

- per-step scheduler latency
- forward-to-forward CPU gap
- SMG gateway overhead
- Python materialization overhead
- output processing overhead
- parser 和 structured output overhead
- 与 vLLM EngineCore、TensorRT-LLM inflight batching 的对比

需要验证：

- 相同模型
- 相同硬件
- 相同 batch/concurrency
- 相同 tool schema
- 相同 prompt distribution
- CPU flamegraph
- scheduler timeline
- GPU idle gap measurement

过度表述风险：

不能只因为 TokenSpeed 有 C++ Scheduler，就声称 CPU 有优势。Agent workloads 仍可能主要受 gateway、tokenizer、parser、output processing 或 Python orchestration overhead 影响。

### Claim 2：TokenSpeed 可能有更强的多轮 agent KV lifecycle control

状态：潜在优势，未验证。

当前证据：

- TokenSpeed 设计中包含 request FSM、page allocation、retract/loadback/writeback 和 KV ownership 概念。
- 如果 KV retention 和 prefix reuse 能与 routing、scheduler decision 协同，这可能有利于多轮 agent workload。

缺失证据：

- tool-call completion 后是否保留 KV
- Round 2 是否能稳定复用 Round 1 prefix KV
- routing 是否保持 KV locality
- KV 在压力下是 evicted、reloaded 还是 recomputed
- 目标模型路径是否支持 hierarchical KVStore
- DeepSeek V4 路径是否支持相关 KV lifecycle features

需要验证：

- Round 1 tool_call + mock tool wait + Round 2 tool_result
- 测量 Round 2 TTFT
- 测量 prefix cache hit rate
- 测量 KV retained / evicted / recomputed / reloaded
- 测量 worker locality hit rate
- 在相同 request distribution 下与 vLLM 和 TensorRT-LLM 对比

过度表述风险：

除非代码路径和 runtime trace 证明，否则不能把 tool-call-specific DDR KV offload 和 same-request resume 描述成已实现能力。

### Claim 3：TokenSpeed 可能更适合 agent-specific scheduling

状态：假设。

当前证据：

- TokenSpeed 有 request FSM 和 execution-plan-oriented design。
- Agent workloads 与普通 chat 不同：更短 burst、更多 round trip、更大的共享 tool prefix、更高的 CPU 固定开销。

缺失证据：

- scheduler policy 是否区分 agent rounds 和普通 chat
- tool schema 或 structured output constraints 是否影响 scheduling decision
- Round 2 requests 是否获得特殊 locality-aware scheduling
- request FSM 是否改善 fairness、tail latency 或 cache retention

需要验证：

- normal chat 与 agent traffic 混合 workload
- P50/P90/P99 latency
- queueing delay
- scheduler decision trace
- per-request cache lifecycle trace

过度表述风险：

除非 scheduler 实际观测并使用 agent-relevant state，否则不能声称 agent-aware scheduling。

### Claim 4：TokenSpeed 可能对 Ascend 适配有战略价值

状态：战略假设，工程风险高。

当前证据：

- TokenSpeed 拆分了 scheduling、model execution 和 gateway concerns。
- 它不像 TensorRT-LLM 那样深度绑定 NVIDIA-only runtime stack。

缺失证据：

- HCCL replacement plan for NCCL assumptions
- CUDA stream/event equivalent behavior
- CUDA graph replacement or fallback strategy
- custom kernel migration plan
- MLA/MoE kernel availability
- memory allocator compatibility
- distributed executor compatibility
- Ascend 上的性能证据

需要验证：

- Ascend 上的最小 decode path
- attention kernel benchmark
- MoE dispatch benchmark
- scheduler-worker communication benchmark
- KV cache allocation and reuse benchmark

过度表述风险：

架构可迁移不代表 kernel 可迁移。Ascend 适配风险集中在通信、内存管理、自定义 kernel 和 graph execution。

### Claim 5：TokenSpeed 可能更适合 DeepSeek-style MLA/MoE workload

状态：未知。

当前证据：

- TokenSpeed 看起来瞄准 Kimi/DeepSeek-style workload 等特定大模型推理路径。
- 它可能包含 model-specific execution 和 kernel integration。

缺失证据：

- 与 vLLM、TensorRT-LLM 的可比 DeepSeek/MLA/MoE benchmark
- 确切 supported model path
- kernel maturity
- TP/EP/DP/CP combination support
- memory footprint comparison
- decode throughput 和 latency comparison
- long-context behavior

需要验证：

- 相同 DeepSeek-style model
- 相同 quantization
- 相同 sequence length
- 相同 batch/concurrency
- 相同 GPU count
- prefill 和 decode 分开测量
- MoE communication counters
- attention/KV memory counters

过度表述风险：

不能从一个优化模型路径泛化到所有 LLM inference workloads。

## 7. 最小 agent benchmark 协议

### 7.1 目标

Benchmark 必须评估 agent-serving 行为，而不能只评估通用 tokens per second。

关键测量项包括：

- CPU control-plane overhead
- forward-to-forward gap
- Round 1 TTFT
- Round 2 TTFT
- tool-call parser overhead
- tool loop overhead
- prefix cache hit rate
- KV retention and eviction
- GPU utilization
- concurrency 下的 tail latency

### 7.2 Workload A：Baseline Chat

目的：

测量无工具场景下的普通 serving overhead。

请求形态：

- system prompt
- one user message
- no tools
- max_tokens = 256

指标：

- TTFT
- TPOT
- output tokens per second
- GPU utilization
- API/server overhead
- scheduler time
- forward-to-forward gap
- CPU utilization

解释：

这是非 agent baseline。任何 agent 优势都必须和这个 baseline 对比。

### 7.3 Workload B：Tool Schema Without Tool Execution

目的：

测量模型没有调用工具时，大 tool schema 的渲染和处理成本。

请求形态：

- system prompt
- user message
- tools list with 8 / 32 / 128 tools
- tool_choice = auto
- prompt 要求模型正常回答，不调用工具

指标：

- chat template rendering time
- tokenizer time
- prompt token count
- prefix cache hit rate
- 相比 Workload A 的 TTFT 增量
- CPU preparation overhead
- memory footprint

解释：

这个 workload 隔离 tool schema 本身带来的 agent overhead，不混入实际 tool call。

### 7.4 Workload C：Two-Round Tool Call

目的：

测量真实 agent loop。

Round 1：

- messages + tools
- model emits tool_call

Gateway/tool loop：

- parse tool_call
- execute mock tool
- append tool_result

Round 2：

- messages + previous assistant tool_call + tool_result
- model emits final answer

指标：

- Round 1 TTFT
- Round 1 TPOT
- tool-call parse latency
- mock tool wait time
- Round 2 request build time
- Round 2 TTFT
- prefix cache hit rate
- KV retained / evicted / recomputed / reloaded
- worker locality hit rate
- end-to-end latency

解释：

这是 agent workload 最重要的 benchmark。如果 TokenSpeed 有优势，它应该体现在 Round 2 TTFT、CPU gap、prefix reuse 或 cache lifecycle behavior 上。

### 7.5 Workload D：Shared Long Prefix Multi-Session Agent Traffic

目的：

测量并发 agent workload 下的 prefix cache 和 routing 行为。

请求形态：

- long shared system prompt
- shared tool schema
- many concurrent sessions
- repeated Round 1 / Round 2 loops
- varied user suffixes

指标：

- prefix cache hit rate
- cache eviction rate
- Round 2 TTFT distribution
- P50/P90/P99 latency
- DP rank skew
- worker locality
- CPU utilization
- GPU utilization
- memory pressure

解释：

这个 benchmark 测试 TokenSpeed 是否能把 shared agent prefixes 转化为真实 serving efficiency。

### 7.6 Workload E：Mixed Normal Chat and Agent Traffic

目的：

测量 agent workload 是否干扰普通 chat workload。

请求形态：

- 70% normal chat
- 30% two-round tool-call traffic
- shared system prompt
- mixed output lengths

指标：

- normal chat TTFT/TPOT
- agent Round 2 TTFT
- scheduler fairness
- queueing delay
- P99 latency
- cache pressure
- GPU utilization

解释：

生产 serving engine 必须能处理 mixed traffic。专门的 agent 优化不应严重伤害普通 chat。

## 8. 必要观测指标

在做强竞争力结论前，至少需要这些 counter 或 trace。

### Gateway / API 层

- request validation time
- chat template rendering time
- tokenizer time
- detokenizer time
- tool parser time
- structured output parser time
- streaming response overhead
- protocol conversion overhead

### Engine / Scheduler 层

- request admission time
- queueing time
- scheduler step time
- cache allocation time
- preemption/retract time
- execution plan build time
- output processing time
- forward-to-forward gap

### KV Cache 层

- prefix cache hit/miss
- KV blocks allocated
- KV blocks retained after request completion
- KV evictions
- KV recomputations
- KV reload/loadback
- host/device KV movement
- cache locality by worker or DP rank

### GPU 层

- forward time
- attention time
- MLP/MoE time
- communication time
- sampling time
- GPU utilization
- memory bandwidth
- HBM usage
- kernel launch overhead

### Distributed 层

- DP routing decision
- TP/PP/EP group configuration
- inter-worker communication time
- all-reduce/all-gather time
- MoE dispatch/combine time
- rank skew

## 9. 安全结论模板

### 9.1 保守结论

基于当前源码级分析，TokenSpeed 在 C++ Scheduler、request FSM 和 KV lifecycle design 上显示出潜在架构差异。但目前还没有足够 benchmark 或 profiling 证据证明它在 agent workload 下整体优于 vLLM 或 TensorRT-LLM。

当前最合适的结论是：TokenSpeed 值得在受控 agent benchmark 下继续评估。

### 9.2 较强但仍安全的结论

TokenSpeed 最有希望的优势不是 generic tool-call support，而是 request FSM、logical KV ownership、scheduler decisions 和 execution-plan materialization 之间更紧密的集成。

如果 profiling 证明它在多轮 agent workload 下有更低的 forward-to-forward CPU gap、更好的 Round 2 prefix reuse 和更低的尾延迟，那么 TokenSpeed 可能在高并发 agent serving 上形成有意义的优势。

### 9.3 否定或谨慎结论

当前证据不足以声称 TokenSpeed 优于 vLLM 或 TensorRT-LLM。TokenSpeed 可能有 promising architecture，但 vLLM 和 TensorRT-LLM 有更成熟的生产生态、更强的可观测性和更已验证的部署路径。

在没有 profiling 和 benchmark 证据前，TokenSpeed 应被描述为 high-potential preview system，而不是已证明更优的 runtime。

### 9.4 不应该说什么

不要说：

- TokenSpeed 已经有比 vLLM 更强的 tool-call support。
- TokenSpeed engine 直接理解 OpenAI 或 Anthropic tool calls。
- Tool wait 会自动触发 DDR KV offload 和 same-request resume。
- DeepSeek V4 hierarchical KVStore 已经是已验证优势。
- C++ Scheduler 自动意味着更低 CPU overhead。
- 没有 kernel-level benchmark 就说 TokenSpeed 优于 TensorRT-LLM。
- 没有公平 benchmark 就说 vLLM 在 agent serving 上很弱。
- SMG gateway capabilities 等同于 TokenSpeed engine capabilities。

### 9.5 推荐定位

TokenSpeed 应被定位为一个高潜力推理运行时。它可能的优势在于调度器、KV 生命周期，以及面向 agent workload 的运行时集成。

当前证据支持进一步评估，但不支持对 vLLM 或 TensorRT-LLM 下最终优越性结论。

## 10. 推荐下一步

1. 为每个主要 TokenSpeed 主张增加源码级引用。
2. 增加一张分离 SMG gateway、TokenSpeed engine、scheduler 和 GPU execution 的运行时图。
3. 增加 scheduler time、cache operations、parser time、forward-to-forward gap 的计数器。
4. 优先实现 Workload C，因为它最接近真实 tool-call agent 行为。
5. 先与 vLLM 对比，再与 TensorRT-LLM 对比。
6. 只有在源码和 benchmark 证据都具备后，才能把潜在优势升级为已验证优势。
