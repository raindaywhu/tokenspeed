# TokenSpeed 架构竞争力横向分析 v0.4

更新时间：2026-05-24

本篇不做 benchmark 计划，而是围绕架构竞争力做横向分析。对比对象收敛为三个：

- TokenSpeed + SMG：基于本仓源码和前面几章分析；
- vLLM / vLLM-Ascend：基于 vLLM V1 官方架构、scheduler、KV cache 文档；
- TensorRT-LLM：基于 NVIDIA 官方 Executor、scheduler、KV cache 文档。

分析框架也收敛为三个问题：

```text
1. 三个框架的整体架构实现有什么差异？
2. 对 agent workload 的 CPU 高负载问题，架构是否能有效解决？
3. 对 agent workload 的 KV 管理问题，架构是否能有效解决？
```

最后再回答：TokenSpeed 是否真的有明显竞争力，还是只是实现风格不同。

## 1. 整体架构实现对比

### 1.1 TokenSpeed：SMG gateway + TokenSpeed engine

TokenSpeed 当前应理解为两层系统：

```text
SMG Gateway
  OpenAI protocol / chat template / tokenizer cache /
  tool parser / reasoning parser / MCP tool loop / worker routing

TokenSpeed Engine
  AsyncLLM / EventLoop / C++ Scheduler /
  KV ownership / MemoryExecutor / ModelExecutor /
  Mapping / CommManager / model forward
```

这个架构的特点是：

1. **Gateway 和 engine 边界清楚**
   `--tool-call-parser`、chat template、MCP tool loop、worker routing 在 SMG；engine 看到的是 tokenized generate request 和 sampling constraints。

2. **Scheduler/KV 是 engine 内部协议**
   TokenSpeed 不是把 KV cache 当成外部插件，而是用 C++ Scheduler 管 request FSM、prefix tree、OwnedPages、tail page、reserve、finish insert、retract/recovery。

3. **执行 loop 保持 Python 灵活性**
   ModelExecutor 和模型 forward 仍在 Python/PyTorch 侧，C++ 主要承担 Scheduler/KV 生命周期。这个选择比纯 C++ runtime 更灵活，但也意味着 CPU overhead 不会天然低于 TensorRT-LLM。

4. **模型并行进入模型层协议**
   attention/dense/MoE 分别有 Mapping 和 process group，DeepSeek V4 走手写模型逻辑 + CommManager，而不是单一 TP/DP 组。

所以 TokenSpeed 的架构假设是：

```text
用 SMG 稳定 agent 请求入口；
用 C++ Scheduler/KV 处理 request 生命周期和 page ownership；
用 Python execution plane 保持模型实现和并行策略灵活性。
```

### 1.2 vLLM V1：API server / EngineCore / GPU worker 多进程架构

vLLM V1 官方文档把架构拆成：

- API server process：处理 HTTP、input processing、tokenization、多模态 loading、streaming；
- Engine Core process：运行 scheduler、管理 KV cache、协调 GPU worker；
- GPU worker process：加载模型、执行 forward、管理 GPU memory；
- DP coordinator：在 data parallel 时做 DP load balancing 和同步 forward。

官方说明 vLLM V1 使用多进程架构来分离职责和提高吞吐，Engine Core 是一个 busy loop，持续调度请求并 dispatch 到 GPU worker。见 vLLM [Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/)。

vLLM V1 的另一个关键点是 unified scheduler：它把 prompt tokens 和 output tokens 统一成 `{request_id: num_tokens}` 这样的 token budget 表达，用同一个 scheduling model 覆盖 chunked prefill、prefix caching、speculative decoding。见 vLLM [V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)。

这说明 vLLM 不是“简单 Python scheduler”。它已经在 V1 中专门重构了 CPU control plane、EngineCore、KV manager 和 worker 边界。

### 1.3 TensorRT-LLM：C++ Executor / in-flight batching / engine-first runtime

TensorRT-LLM 的架构更偏 backend/runtime：

- Model Definition API 把模型构造成可编译的 TensorRT engine；
- C++ runtime / C++ backend 是推荐路径；
- Executor API 支持 async request execution 和 in-flight batching；
- PyExecutor / C++ backend loop 内部包含 Scheduler、KVCacheManager、ModelEngine、Sampler；
- runtime optimization 包括 CUDA Graph、Overlap Scheduler、speculative decoding 等。

官方 Executor API 文档说明 TensorRT-LLM 提供高层 C++ API，可异步执行请求并支持 in-flight batching。见 [Executor API](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html)。官方架构文档也说明 C++ backend 实现 in-flight batching，是推荐 backend。见 [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/0.19.0/architecture/overview.html)。

因此 TensorRT-LLM 的架构假设和 TokenSpeed 不同：

```text
把模型和 runtime 尽量编译/固化到高性能 NVIDIA backend；
用 Executor + in-flight batching + CUDA Graph / Overlap Scheduler 压低 CPU/GPU 开销。
```

### 1.4 整体架构判断

| 维度 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| 入口层 | SMG gateway，agent/tool/parser 能力较完整 | OpenAI API server，多进程 input/output 处理 | LLM API / Triton backend / Executor |
| Scheduler 位置 | TokenSpeed engine 内，C++ FSM | EngineCore busy loop | Executor / PyExecutor / C++ runtime |
| KV 管理 | C++ Scheduler + KV ownership + prefix tree | KVCacheManager + block pool + prefix cache | KVCacheManager + paged KV + reuse/offload/event API |
| 执行层 | Python ModelExecutor + custom kernels/backend | GPU worker + PyTorch/custom kernels | TensorRT engine + C++ runtime |
| CPU overhead 方向 | C++ Scheduler + EventLoop overlap | 多进程 EngineCore + persistent batch + CUDA graphs | C++ Executor + CUDA Graph + Overlap Scheduler |
| 灵活性 | 高，模型实现留在 Python | 高，模型生态强 | 中，engine/backend 绑定更强 |

初步判断：

- TokenSpeed 在架构上不是 vLLM 的简单变体，但也不是全面领先。
- vLLM V1 已经非常重视 CPU control plane 和 KV cache manager。
- TensorRT-LLM 在 NVIDIA backend 上对 CPU overhead 和 runtime optimization 的工程成熟度更强。
- TokenSpeed 的竞争力必须落到更具体的问题：agent short-decode 下的 CPU 高负载，以及多轮 prefix 下的 KV lifecycle。

## 2. Agent workload 问题一：CPU 高负载

### 2.1 问题定义

Agent workload 的 CPU 高负载通常来自：

- OpenAI protocol / messages / tools / chat template；
- tokenization / detokenization / streaming；
- structured output / tool parser；
- 短 decode step 下 scheduler 每步频繁运行；
- output feedback、stop 判断、abort/finish cleanup；
- 多 DP / TP 场景下 metadata 同步和 worker dispatch；
- tool loop 后下一轮 request 构造。

关键不是“CPU 使用率高”本身，而是 CPU 控制面是否拖住 GPU：

```text
decode step 越短，GPU forward 越快，
CPU scheduler/output/parser 的固定开销越容易变成 p95 ITL。
```

### 2.2 TokenSpeed 如何处理

TokenSpeed 有三层应对：

1. **SMG 分担入口控制面**
   SMG 处理 chat template、tokenizer cache、tool/reasoning parser、MCP tool loop。这样 engine 不需要理解 OpenAI messages/tools 原始结构。

2. **C++ Scheduler 承担 request/KV 决策**
   admission、prefix match、page ownership、finish/retract 等高频状态逻辑在 C++ Scheduler 内，不是纯 Python 数据结构操作。

3. **EventLoop overlap**
   EventLoop 将 cache op、forward op、output feedback、scheduler.advance 串成闭环，当前 forward 可以和上一轮 output commit / cache state 更新错开。

这套设计对 CPU 高负载有价值，尤其是短 decode、多轮 request churn 场景。但它不是天然碾压：模型 forward、input buffer、distributed metadata、部分 output processing 仍在 Python runtime 内。

### 2.3 vLLM 如何处理

vLLM V1 对 CPU overhead 的设计非常明确。vLLM 官方 V1 博客写到，GPU 变快后，API server、scheduler、input preparation、detokenization、streaming 的 CPU overhead 更明显；V1 通过 multiprocessing API server、隔离 EngineCore execution loop 来让 tokenization、多模态 processing、detokenization、streaming 与核心 scheduler/model executor overlap。见 [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)。

vLLM V1 还提到：

- simple unified scheduler 用 token budget 表达调度；
- persistent batch 缓存 input tensors，只应用 step diff，减少每步重建 metadata；
- prefix caching 数据结构降低 Python object 创建和 eviction 开销；
- chunked prefill 默认优先 decode，在 token budget 允许时再放 prefill，以改善 ITL。见 vLLM [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)。

所以，在 CPU 高负载问题上，vLLM V1 已经是强对手。TokenSpeed 不能只靠“有 C++ scheduler”就得出明显优势。

### 2.4 TensorRT-LLM 如何处理

TensorRT-LLM 对 CPU 高负载的处理更 backend-first：

- C++ Executor API 支持 async execution 和 in-flight batching；
- C++ backend 是推荐 serving backend；
- CUDA Graph 减少 CPU-side kernel launch overhead；
- Overlap Scheduler 将下一 step GPU work 提前启动，同时 CPU 处理上一 step 的 stop / response update。

NVIDIA 官方架构文档明确说明 Overlap Scheduler 的策略是：不等 CPU 处理完第 n 步结果，就启动第 n+1 步 GPU work，从而把 CPU-bound latency 隐藏在 GPU computation 后面；该 scheduler 默认开启。见 TensorRT-LLM [Architecture Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)。

这意味着如果只看 CPU 高负载 / GPU idle gap，TensorRT-LLM 在 NVIDIA 平台上是非常强的 baseline。

### 2.5 CPU 高负载横向结论

| 框架 | 是否有效解决 CPU 高负载 | 判断 |
|---|---|---|
| TokenSpeed | 部分有效 | C++ Scheduler + EventLoop overlap 有价值，但 Python execution plane 仍存在 |
| vLLM V1 | 有效 | V1 明确围绕 CPU overhead 重构 EngineCore、API server、persistent batch |
| TensorRT-LLM | 很强 | C++ Executor、CUDA Graph、Overlap Scheduler 对 CPU/GPU gap 更直接 |

结论：

```text
CPU 高负载不是 TokenSpeed 的绝对优势点。
它有竞争力候选，但 vLLM V1 和 TensorRT-LLM 都已经系统性解决这个问题。
TokenSpeed 只有在 C++ Scheduler/KV feedback 与 EventLoop overlap 组合
显著降低 agent short-decode p95 时，才能说有差异。
```

因此报告里不应写：

```text
TokenSpeed 有 C++ Scheduler，所以 CPU overhead 显著优于 vLLM/TensorRT。
```

更合理的写法是：

```text
TokenSpeed 在 CPU control plane 上有明确设计，但横向看不是唯一有解。
vLLM V1 也在降低 CPU overhead，TensorRT-LLM 在 NVIDIA backend 上更激进。
TokenSpeed 的差异要和 KV lifecycle / output feedback 绑定起来看。
```

## 3. Agent workload 问题二：KV 管理

### 3.1 问题定义

Agent workload 的 KV 管理压力不是普通长上下文那么简单，而是：

- 多轮 messages/tools/history 形成 repeated prefix；
- tool call 后下一轮 request 需要复用前序上下文；
- finish/abort 很频繁，KV block/page 生命周期复杂；
- 短 decode 造成 tail page waste；
- KV 接近满载时需要 preempt / retract / evict / offload；
- 多 replica 时，KV locality 和 load balance 冲突；
- host/device/offload cache 是否能被 scheduler 正确利用。

关键问题是：

```text
KV 不只是 block cache，而是 request lifecycle 的一部分。
谁拥有 page？什么时候能复用？什么时候该释放、回收、下沉、恢复？
```

### 3.2 TokenSpeed 如何处理

TokenSpeed 的 KV 管理最值得研究。根据本地代码分析，它把 KV page ownership 放进 C++ Scheduler 的 request FSM：

- prefix tree / safe reuse；
- request-local tail page；
- dynamic decode reserve；
- finish insert；
- retract / recovery；
- cache op + forward op 同在 ExecutionPlan；
- scheduler.advance 根据 output feedback 推进状态；
- DP 场景下还会同步 global token / batch / mode metadata。

这和“有 prefix cache”不是一回事。TokenSpeed 试图把：

```text
request state
  + KV page ownership
  + prefix tree
  + cache movement
  + output feedback
```

绑定成一个 runtime protocol。

它的强点是安全复用和生命周期语义；弱点是当前 DeepSeek V4 path 明确禁用 hierarchical KVStore，不能把 host/L2/L3 offload 直接算作 V4 当前收益。

### 3.3 vLLM 如何处理

vLLM 的 KV 管理也很成熟。官方 prefix caching 设计说明，vLLM V1 的 prefix cache 实现在 KV cache manager 中，核心 `KVCacheBlock` 包含 immutable block id、block hash、ref count 和 free queue 指针；KV cache manager 初始化时预分配 block pool，并维护 free queue、hash key 到 block id、request id 到 block id 的映射。见 vLLM [Automatic Prefix Caching](https://docs.vllm.ai/en/v0.11.1/design/prefix_caching/)。

vLLM 的调度路径中，scheduler 会调用：

- `get_computed_blocks()` 查找已计算 prefix；
- `allocate_slots()` 分配新 block、touch 已命中 block、从 free queue 分配或 evict；
- running request 会 append token ids 到 slots，block full 后进入 cache。

vLLM V1 的特点是 hash-based prefix cache + LRU eviction + block pool，且 V1 文档强调 prefix cache 低 overhead。vLLM 还有 Hybrid KV Cache Manager，用 KVCacheCoordinator 管理不同 KV cache group，例如 full attention + sliding window / Mamba 等组合。见 vLLM [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)。

另外，vLLM 的 disaggregated prefill 通过 prefill instance、decode instance 和 KV connector 转移 KV cache，官方文档说明 connector / LookupBuffer / Pipe 是核心抽象。见 vLLM [Disaggregated Prefilling](https://docs.vllm.ai/en/v0.12.0/features/disagg_prefill/)。

因此 vLLM 不是弱 KV baseline。它已经有：

- paged KV；
- block pool；
- prefix cache；
- hybrid KV group；
- disaggregated prefill KV transfer；
- preemption/recompute 策略；
- KV event / connector 相关机制。

TokenSpeed 如果要证明 KV 管理更强，不能只说“prefix cache”。它必须证明 request FSM + page ownership + finish/retract 语义更紧密。

### 3.4 TensorRT-LLM 如何处理

TensorRT-LLM 的 KV 管理同样很强，而且在一些方面比 TokenSpeed 当前落地更完整。

官方 KV cache 文档说明 TensorRT-LLM 的 KV cache 支持跨请求 reuse，并提供 offloading、prioritized eviction 等工具来增加 reuse。见 TensorRT-LLM [KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)。

TensorRT-LLM legacy KV cache reuse 文档说明：

- KV cache pages 可以被相同 prompt 开头的请求共享；
- reuse 可降低 first token latency；
- KV cache reuse 默认可通过 KVCacheManager / `enableBlockReuse` 启用；
- KV state 只有在计算该 state 的请求终止后才可复用；
- reusable blocks 内存不足时按 LRU evict；
- host memory offload 可以延长 reusable blocks 存活时间，但会引入 CPU/GPU copy 成本。见 [KV cache reuse](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md)。

更重要的是，TensorRT-LLM 还有 KV cache event API。NVIDIA 技术博客说明 Executor API 暴露 KV cache updates，block stored / removed / updated 时会发事件；这些事件可被单 executor 或多 executor 聚合，用于 KV-aware routing 和 scheduling。见 NVIDIA [KV cache reuse optimizations blog](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)。

TensorRT-LLM scheduler 文档也说明 CapacityScheduler 会根据 KV cache capacity 等资源决定哪些 active request 能被分配资源，并支持 paused_requests。见 TensorRT-LLM [Scheduler](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html)。

这说明 TensorRT-LLM 在 KV 管理上不是只有 paged cache，而是有：

- block reuse；
- host offload；
- priority retention / eviction；
- KV event API；
- KV-aware routing；
- scheduler resource capacity 判断；
- C++ runtime pause 支持。

### 3.5 KV 管理横向结论

| 框架 | KV 管理能力 | 对 agent 多轮 prefix 的适配性 | 判断 |
|---|---|---|---|
| TokenSpeed | request FSM + KV ownership + prefix tree + finish/retract | 理论上很贴合，但 V4 hierarchical KVStore 当前禁用 | 强候选，但不能夸大 |
| vLLM V1 | block pool + hash prefix cache + scheduler allocation + hybrid KV manager | 成熟，生态强，APC 已系统化 | 强 baseline |
| TensorRT-LLM | KV reuse + host offload + priority eviction + KV event API | 非常强，尤其适合 KV-aware routing/offload | 强 baseline，NVIDIA 平台优势明显 |

结论：

```text
KV 管理是 TokenSpeed 最可能形成竞争力的方向，
但不是因为“别人没有 KV cache”。
vLLM 和 TensorRT-LLM 都有成熟 KV 管理。
TokenSpeed 的差异只在于：它是否把 request FSM、KV ownership、
finish/retract/recovery 和 execution plan 绑定得更深。
```

换句话说，TokenSpeed 的 KV 竞争力是“生命周期协议”竞争力，不是“缓存功能”竞争力。

## 4. TokenSpeed 是否有明显竞争力

### 4.1 不能成立的竞争力说法

这些说法都不够严谨：

```text
TokenSpeed 有 C++ Scheduler，所以 CPU overhead 一定优于 vLLM。
TokenSpeed 有 prefix cache，所以 agent 场景一定更强。
TokenSpeed 有 SMG tool parser，所以 agent runtime 有护城河。
TokenSpeed 有 MLA kernel，所以适配价值很高。
TokenSpeed 支持 EP，所以 MoE serving 有优势。
```

横向看，这些单点都站不稳：

- vLLM V1 也在系统性降低 CPU overhead；
- TensorRT-LLM 在 C++ Executor、CUDA Graph、Overlap Scheduler 上更激进；
- vLLM / TensorRT-LLM 都有成熟 KV cache / prefix reuse；
- TensorRT-LLM 还有 host offload、priority eviction、KV event API；
- parser 和 single kernel 都容易被复制或已经被吸收；
- EP 本身不是优势，关键是 dispatch-combine 和 communication protocol。

### 4.2 能成立的竞争力假设

TokenSpeed 真正有竞争力的假设应该写成：

```text
对于 agentic MoE serving，
TokenSpeed 的竞争力不来自单个功能，
而来自 SMG gateway、C++ Scheduler/KV ownership、EventLoop feedback、
split parallelism 和 model backend 的组合协议。
```

其中最值得保留的三条：

1. **Scheduler / KV ownership**
   如果 TokenSpeed 的 request FSM + page ownership 确实比 vLLM/TensorRT 更紧密地管理 finish/retract/recovery，那么它在多轮 prefix + KV pressure 场景有竞争力。

2. **EventLoop 与 KV feedback**
   如果 cache op / forward op / output feedback / scheduler.advance 的闭环能降低短 decode control-plane gap，那么它比普通 async output 更有价值。

3. **split parallelism 作为 model-runtime 协议**
   如果 attention/dense/MoE 不同 execution domain 真的减少了 communication 和 MoE token movement，那么它是 MoE serving 的竞争力候选。

### 4.3 当前结论：局部有竞争力，但不是全面领先

横向比较后，当前结论应更克制：

| 问题 | TokenSpeed 相对 vLLM | TokenSpeed 相对 TensorRT-LLM | 结论 |
|---|---|---|---|
| 整体架构 | 更强调 C++ Scheduler/KV + Python model flexibility；vLLM V1 架构也很成熟 | 更灵活，但 backend 优化不如 TensorRT-LLM 固化 | 不是全面领先 |
| CPU 高负载 | vLLM V1 已做 EngineCore、多进程、persistent batch，TokenSpeed 优势不明显 | TensorRT-LLM C++ Executor + Overlap Scheduler 更强 | TokenSpeed 不构成明显优势 |
| KV 管理 | TokenSpeed lifecycle 语义可能更深，但 vLLM KV manager 很成熟 | TensorRT-LLM reuse/offload/event API 很强 | TokenSpeed 是强候选，但需谨慎 |
| Agent gateway | SMG 比普通 backend 更贴近 agent tool loop | TensorRT-LLM 更偏 engine/backend，不是 gateway runtime | TokenSpeed 有中等优势 |
| 并行策略 | TokenSpeed layer-family split 更值得研究 | TensorRT-LLM backend/MoE 优化强，但硬件绑定 | TokenSpeed 在架构表达上有候选优势 |

最终判断：

```text
TokenSpeed 不是在所有维度明显强于 vLLM/TensorRT-LLM。
在 CPU 高负载问题上，它不是明显领先者；
在 KV 管理问题上，它有架构级竞争力候选；
在 agent gateway + KV lifecycle + split parallelism 组合上，
它可能形成区别于 vLLM/TensorRT-LLM 的系统性价值。
```

因此报告的表述应该是：

```text
TokenSpeed 值得研究，不是因为它有别人没有的单点功能，
而是因为它试图把 agent gateway、request FSM、KV ownership、
short-decode execution loop 和 MoE split execution domain 绑成一个 runtime。

但横向看，vLLM V1 和 TensorRT-LLM 已经分别在 CPU overhead 和 KV cache 上很强。
TokenSpeed 的竞争力应被定义为“局部架构竞争力候选”，
核心看 Scheduler/KV lifecycle 与并行 execution protocol 是否能形成不可小 patch 复制的差异。
```

## 5. TokenSpeed 内部机制回填

横向对比容易把 TokenSpeed 讲薄，因为 vLLM 和 TensorRT-LLM 都能列出类似功能名：prefix cache、paged KV、scheduler、EP、CUDA graph、overlap。真正要判断 TokenSpeed 是否有竞争力，必须回到它自己的实现耦合方式。

### 5.1 Scheduler/KV ownership：不是 cache feature，而是 request lifecycle protocol

TokenSpeed 的关键不是“它有 KV cache”，而是 C++ Scheduler 在一次 `schedule -> cache op -> forward op -> output feedback -> advance` 闭环里同时管理 request 状态和 KV page ownership。

从前面源码分析看，Scheduler 不只是决定下一批请求跑多少 token，还同时处理：

- prefix tree match；
- request active page 和 cached page 的归属；
- request-local tail page；
- decode reserve；
- finish 后把生成内容插回 prefix tree；
- abort/finish/retract/recovery 对 page 的释放或恢复；
- scheduler.advance 根据 output feedback 推进 request 状态；
- ExecutionPlan 同时携带 cache movement 和 forward metadata。

这意味着 TokenSpeed 的 KV 管理更像一套生命周期协议：

```text
Request FSM
  -> Prefix match
  -> Page ownership
  -> Cache operation
  -> Forward operation
  -> Output feedback
  -> Finish / retract / recovery
  -> Prefix tree update
```

这个设计解决的是 agent workload 下最容易被低估的问题：请求不是稳定的一条长序列，而是频繁 finish、tool call、下一轮 resume、abort、重入。KV page 如果只是被当成 block pool，很容易把“能不能复用”简化成 hash 命中；但实际更难的是“什么时候可以安全复用，什么时候必须释放，什么时候可以恢复”。

对 vLLM 的复制判断也要因此更细：

| 复制层级 | 难度 | 说明 |
|---|---|---|
| prefix hash / block pool | 低到中 | vLLM 已经有成熟 APC，不是 TokenSpeed 独有 |
| finish 后复用生成内容 | 中 | 需要保证 output feedback 与 cache 状态一致 |
| request FSM + page ownership 一体化 | 高 | 需要改 scheduler/KV manager 的状态边界 |
| retract/recovery 与 execution plan 绑定 | 高 | 不只是 cache eviction，而是运行时状态恢复 |

所以，TokenSpeed 在 KV 方向的竞争力必须这样表述：

```text
不是“TokenSpeed 有 prefix cache”，而是“TokenSpeed 试图把 prefix reuse、
page ownership、finish/retract/recovery 变成 scheduler 的一等语义”。
```

### 5.2 EventLoop：短 decode 场景的控制面闭环

Agent workload 常见形态是：

```text
long prefill
  -> short decode
  -> tool call
  -> append tool result
  -> next short decode
```

短 decode step 下，单步 GPU forward 可能很快，CPU 侧的 scheduler、output processing、stop 判断、metadata 更新、下一步 batch 构造反而会暴露出来。TokenSpeed 的 EventLoop 价值在于它不是把这些动作串成一个单线程同步路径，而是把 cache operation、forward operation、output feedback 和 scheduler.advance 拆开形成闭环。

这类设计影响的不是一个单独 kernel，而是 p95 ITL：

- 当前 step 的 forward 可以和上一轮 output commit / state advance 有机会错开；
- cache op 和 forward op 同在 ExecutionPlan，KV movement 不再是 executor 外部副作用；
- output feedback 不是最终日志，而会反向推进 Scheduler 的 request FSM；
- DP 下还要同步 global token / batch / mode metadata，避免 idle rank 破坏 collective progress。

对比 vLLM 和 TensorRT-LLM时，这个点不能被夸大。vLLM V1 已经有 EngineCore busy loop、persistent batch 和 async output processing；TensorRT-LLM 的 Overlap Scheduler 在 NVIDIA 平台上更直接地隐藏 CPU stop/response update。TokenSpeed 的差异只在于：它的 EventLoop 是否和 KV ownership / ExecutionPlan 绑定得更深。

因此可以给出一个更精确的判断：

```text
如果 workload 主要是大 batch 长 decode，TokenSpeed EventLoop 未必明显领先；
如果 workload 是高 churn、多轮、短 decode、KV pressure，
EventLoop + Scheduler/KV feedback 才可能移动 p95 ITL。
```

### 5.3 并行策略：不是支持 EP，而是 layer-family execution domain

并行策略这部分也容易讲浅。`--enable-expert-parallel`、`--data-parallel-size`、`--tensor-parallel-size` 这些能力本身各家都有，不能构成竞争力。TokenSpeed 值得研究的是它是否把不同 layer family 的最优并行形态明确拆开。

可以把它理解为：

```text
attention domain: latency-sensitive, head/KV layout-sensitive
dense domain: GEMM throughput-sensitive
MoE domain: expert placement, token dispatch, all-to-all-sensitive
DP domain: request/session/KV locality-sensitive
```

如果这些 domain 被迫共用一个 TP/DP/EP 组合，就会出现结构性浪费：

- attention decode 本来 token 很少，被过度 TP 后通信延迟可能盖过计算收益；
- dense/shared expert 可能需要 TP 分摊 GEMM；
- routed MoE 更依赖 EP 和 dispatch-combine，而不是简单 dense TP；
- 多 DP replica 下，rank token count 和 session KV footprint 会 skew，communication 不能假设各 rank token 一样。

前面源码分析里，TokenSpeed 的 `Mapping`、process group、`CommManager`、MoE backend 和 placement compiler 都应该放在这个问题下讲，而不是各自散讲：

| 实现点 | 应回答的问题 | 竞争力边界 |
|---|---|---|
| split Mapping | attention/dense/MoE 是否能用不同 group | 配置名容易复制，真正难点是贯穿模型 forward |
| CommManager | hidden state 何时 AR/AG/RS，是否 token-aware | 需要和真实 token distribution、residual placement、CUDA graph shape 协同 |
| MoE backend | expert weight、top-k token、dispatch-combine 如何组织 | 硬件绑定强，Ascend 迁移需要重做等价 backend |
| placement compiler | 手写通信规则能否系统化 | 工程护城河大于当前 V4 直接性能收益 |

这也是为什么并行策略不能写成“TokenSpeed 支持 EP”。更准确的说法是：

```text
TokenSpeed 的并行策略竞争力候选在于：
它试图把 attention、dense、MoE、DP/TP/EP 的 execution domain
作为模型 runtime 协议表达，而不是只靠启动参数拼后端能力。
```

但边界也必须保留：

- 当前 DeepSeek V4 路径更像手写模型逻辑 + `CommManager` + 自定义 attention/MoE，不应说已完全依赖 generic placement compiler；
- TokenSpeed 当前仍有限制，例如 MoE TP 和 EP 不能随意同时放大；
- Ascend 上要复现这类收益，真正难点在 CANN/HCCL/自研 MoE dispatch-combine，而不是参数接口。

### 5.4 local-SPMD / placement compiler：服务并行策略，不是独立卖点

local-SPMD / placement compiler 在报告中的位置应该后移：它不是并行策略的全部，也不是 DeepSeek V4 当前主性能来源，而是让复杂并行策略可维护、可扩展的机制。

它的工作原理可以讲成三层：

```text
Placement annotation
  表达 tensor/module 边界上的 Replicate / Shard / Partial / ATTN_TP / DENSE_TP / MOE_TP_EP

Static compiler
  沿 decoder layer module spec 推导 placement 变化

Collective insertion
  在 placement 不匹配处插入 all-gather / reduce-scatter / all-reduce /
  deferred reduce / residual slice / fused reduce-norm
```

这解决的不是“某次 forward 更快一点”，而是：

```text
当模型越来越像 V4/Kimi：attention、dense、routed expert、shared expert、
latent-KV、DP/TP/EP/CP 混在一起时，手写通信规则很难长期维护。
```

对 vLLM 复制难度的判断：

- 如果只是支持更多 parallel size 参数：配置级复制；
- 如果模型内部按 layer family 走不同 process group：model patch 级复制；
- 如果 placement、weight loader、forward hidden-state movement、collective insertion 统一表达：架构级复制。

所以 placement compiler 的价值应该写成：

```text
它不直接证明 TokenSpeed 当前更快；
它证明 TokenSpeed 在复杂 MoE/latent-KV 并行策略上有系统化落地路径。
```

## 6. 性能归因账本

这份报告仍然需要回答“这些设计会让哪些性能指标移动”。但写法应该是机制归因，而不是把多个百分比线性相加。

| 机制 | 主要改变的瓶颈 | 应该移动的指标 | 为什么 |
|---|---|---|---|
| SMG tokenizer/cache | gateway prompt preparation | TTFT、CPU time/request | repeated template/tokenization 不再每轮重做 |
| SMG worker routing | worker KV locality / load skew | TTFT、queue p95、prefix hit | 多轮 session 更可能落到已有 cache 的 worker |
| Scheduler prefix reuse | repeated prefill compute | TTFT、prefill tokens saved | cached prefix 减少重复 prefill |
| finish insert | 下一轮可复用范围 | TTFT、prefix hit after tool loop | 生成内容安全进入 prefix tree |
| request-local tail page | KV fragmentation | free page watermark、p95 queue | 短 decode 尾页浪费下降 |
| decode reserve | KV admission 稳定性 | stall 次数、p95 ITL | accepted tokens 变化不至于造成 page 抖动 |
| retract/recovery | KV 满载恢复成本 | p99 queue、recompute/offload 次数 | 比粗暴失败或全量重算更细粒度 |
| EventLoop overlap | CPU/GPU control gap | GPU idle gap、p95 ITL | output feedback 与下一步执行重叠 |
| split parallelism | 不合适的统一 TP/EP | collective bytes、MoE layer time | attention/dense/MoE 分别选 group |
| token-aware communication | rank token skew / padding waste | per-rank token skew、comm time | 按真实 token count 通信 |
| MoE EP backend | expert dispatch-combine | all-to-all bytes、expert GEMM util | 更贴近 routed expert 瓶颈 |
| placement compiler | 并行策略实现成本 | 新模型适配周期、通信 bug | 静态插入 collective，减少手写规则 |
| MLA / latent-KV | KV layout / decode memory bandwidth | decode kernel time、KV bandwidth | 价值在端到端 layout，不在单 kernel |

综合性能模型应写成瓶颈迁移：

```text
如果 TTFT 主要来自重复 prefill，Scheduler prefix reuse / finish insert 更重要；
如果 p95 ITL 主要来自 CPU/GPU gap，EventLoop / persistent batch / overlap scheduler 更重要；
如果 MoE layer time 占比高，split parallelism / EP backend / dispatch-combine 更重要；
如果 KV capacity 接近上限，page ownership / retract / offload / eviction 更重要。
```

因此不能写：

```text
Scheduler/KV 20% + split parallelism 20% + MLA 20% = 总收益 60%。
```

应该写：

```text
这些优化会在不同 workload 区间接管主瓶颈；
最终收益取决于 agent 多轮 prefix 比例、decode step 长度、MoE token skew、
KV pressure 和目标硬件 collective/kernel 效率。
```

## 7. 报告应该如何组织

正式 PPT/报告建议按这个结构讲：

1. **整体架构实现对比**
   - TokenSpeed：SMG + C++ Scheduler/KV + Python execution；
   - vLLM：API server + EngineCore + GPU workers；
   - TensorRT-LLM：C++ Executor + TensorRT engine + IFB。

2. **Agent workload 问题一：CPU 高负载**
   - TokenSpeed 的 C++ Scheduler/EventLoop；
   - vLLM V1 的 EngineCore / persistent batch；
   - TensorRT-LLM 的 CUDA Graph / Overlap Scheduler；
   - 结论：TokenSpeed 不是明显领先。

3. **Agent workload 问题二：KV 管理**
   - TokenSpeed 的 request FSM + KV ownership；
   - vLLM 的 KVCacheManager / APC / hybrid cache；
   - TensorRT-LLM 的 KV reuse / offload / event API；
   - 结论：TokenSpeed 有强候选，但要限定为 lifecycle protocol。

4. **最终竞争力判断**
   - 单点功能不够；
   - CPU 高负载不是明显优势；
   - KV lifecycle 是最大候选；
   - split parallelism 是第二候选；
   - MLA/kernel 比重降低。

## 8. 外部资料索引

- vLLM V1 architecture: https://docs.vllm.ai/en/stable/design/arch_overview/
- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide/
- vLLM V1 blog: https://vllm.ai/blog/2025-01-27-v1-alpha-release
- vLLM prefix caching: https://docs.vllm.ai/en/v0.11.1/design/prefix_caching/
- vLLM hybrid KV cache manager: https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- vLLM disaggregated prefill: https://docs.vllm.ai/en/v0.12.0/features/disagg_prefill/
- TensorRT-LLM architecture: https://nvidia.github.io/TensorRT-LLM/architecture/overview.html
- TensorRT-LLM Executor API: https://nvidia.github.io/TensorRT-LLM/advanced/executor.html
- TensorRT-LLM scheduler: https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html
- TensorRT-LLM KV cache system: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
- TensorRT-LLM KV cache reuse: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md
- NVIDIA KV cache reuse optimizations: https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/
