# TokenSpeed Ascend / Agent Runtime 技术解读

更新时间：2026-05-21

本目录整理的是一组面向 **TokenSpeed 架构评估与 Ascend 910C + 950DT 适配判断** 的源码分析文档。它不是官方使用手册，也不是功能清单，而是回答一个更尖锐的问题：

> TokenSpeed 的核心能力到底是如何实现的？这些能力对 DeepSeek V4 / Kimi 类 MoE 与 agentic workload 有什么价值？vLLM / vLLM-Ascend 是不是容易迁移复制？

此前提交的版本过于像英文摘要，缺少系统架构、请求流、KV 生命周期和并行执行协议的展开。这一版重新按“技术解读”组织：先讲 TokenSpeed 是什么样的运行时系统，再用一条请求在系统内的流动把 SMG gateway、Scheduler/KV、并行策略、placement compiler 和性能收益挂起来。

## 一句话结论

TokenSpeed 不应被理解成“带 C++ scheduler 和 MLA kernel 的 vLLM 类 serving 后端”。更准确的理解是：

```text
TokenSpeed + SMG 是一套把 agent-facing gateway、请求生命周期、
KV page ownership、cache movement、模型层并行拓扑和 GPU forward
物化连接成闭环的推理运行时系统。
```

它最值得研究的地方不是单点功能，而是几个运行时协议是否形成了架构级优势：

- C++ Scheduler 把 request FSM 和 KV page ownership 绑在一起；
- Python EventLoop 把 cache op、forward op、output feedback 串成可 overlap 的执行闭环；
- ModelExecutor 把 C++ 的逻辑 page plan 物化为 GPU 侧 `req_to_page`、`InputBuffers`、`ForwardContext`；
- Mapping / CommManager / placement compiler 把 attention、dense、MoE 的不同并行域带进模型执行；
- DeepSeek V4 路径进一步加入 latent/compressed KV、paged cache group、MoE backend 和 token-aware communication。
- SMG gateway 承担 OpenAI protocol、chat template、tokenizer cache、tool/reasoning parser、MCP tool loop 和 worker routing，为 engine 侧 KV reuse 提供稳定输入和路由前提。

## 研究目标

这组文档服务于 TokenSpeed 架构评审、Ascend 适配判断和技术报告/PPT 内容设计，重点回答五个问题：

1. **TokenSpeed 具体怎么实现？**  
   从进程、C++/Python 边界、CPU/GPU 边界、rank/DP/TP 边界、请求状态和 KV ownership 逐层展开。

2. **一个请求在 TokenSpeed 里怎么执行？**  
   用 generate 请求为线索，串起 `AsyncLLM -> RequestHandler -> EventLoop -> C++ Scheduler -> MemoryExecutor / ModelExecutor -> OutputProcessor -> Scheduler.advance`。

3. **Agent 场景下 KV 管理优化在哪里？**  
   重点不是“有 prefix cache”，而是 prefix reuse、request-local tail page、decode reserve、finish insert、retract/loadback/writeback 都进入 C++ request FSM。

4. **并行策略特别在哪里？**  
   重点不是“支持 EP”，而是 attention / dense / MoE 可以拥有不同 group 和通信边界，并通过 `Mapping`、`CommManager` 或 placement compiler 影响真实 hidden-state movement。

5. **vLLM / vLLM-Ascend 是否容易复制？**  
   每个机制都要区分：config 可复制、model patch 可复制、runtime protocol 可复制，还是需要架构级重构。

## 阅读顺序

1. [TokenSpeed Runtime 架构纠偏](01-runtime-architecture.md)  
   解释 TokenSpeed 的五个运行时面：Main Python、DP Controller、Scheduler Worker、嵌入式 C++ Scheduler、GPU execution plane。

2. [TokenSpeed 请求上下文执行图谱](02-request-lifecycle.md)  
   用一条 generate 请求说明系统如何接入、admission、调度、占用 KV、执行 forward、输出 token 并反向推进 scheduler。

3. [Agent 场景 KV 管理优化深挖](03-agent-kv-management.md)  
   深入说明 KV page ownership、prefix tree、request-local allocator、decode reserve、finish/retract/writeback/loadback 如何服务多轮 agent。

4. [并行策略 / Placement Compiler 深挖](04-parallel-strategy-and-placement.md)  
   解释 split parallelism 不是 CLI 参数堆叠，而是如何落到 Mapping、process group、CommManager、placement compiler 和模型 forward。

5. [架构竞争力横向分析](05-architecture-competitiveness.md)
   横向对比 TokenSpeed、vLLM V1、TensorRT-LLM 的整体架构、agent CPU 控制面、KV 管理能力，再回到 TokenSpeed 内部机制判断 Scheduler/KV、EventLoop、split parallelism 和 placement compiler 是否构成架构级竞争力。

6. [代码地图与未闭合问题](06-code-map-and-open-questions.md)  
   列出当前已经读过的源码入口、已经确认的判断边界，以及下一步还需要继续验证的关键问题。

7. [SMG Gateway 技术拆解](07-smg-gateway-runtime.md)
   单独拆解 SMG 的进程边界、gateway request pipeline、tool/reasoning parser、MCP tool loop、worker routing，以及它和 TokenSpeed engine 的职责切分。

## 当前判断边界

为了避免技术判断过度包装，当前结论保留这些边界：

- **不应声称 TokenSpeed 有 tool-call 专用 DDR KV offload。**  
  当前读到的 tool call 主要表现为 chat template / stop token / normal finish path，没有看到 `ToolCallEvent -> offload this request KV to DDR` 的专门策略。

- **不应把 SMG gateway 行为写成 TokenSpeed engine 行为。**
  `--tool-call-parser`、chat template、tokenizer cache、MCP tool loop 和 OpenAI-compatible response formatting 属于 SMG；engine 侧主要接收 tokenized generate request 与 sampling constraints。

- **不应把 DeepSeek V4 hierarchical KVStore 当成已落地收益。**  
  当前分支仍要求 DeepSeek V4 baseline 使用 `--disable-kvstore`，`DeepseekV4TokenToKVPool` 的 offload / reload 路径也明确未实现。

- **不应声称 DeepSeek V4 已完全走 generic placement compiler。**  
  DeepSeek V4 当前实际路径更像手写模型逻辑 + `CommManager` + 自定义 attention/MoE；placement compiler 是通用基础设施，不能直接等同于 V4 主路径。

- **不应把 MLA kernel 作为最大护城河。**  
  vLLM 主线已经有 MLA / DeepSeek V4 相关 KV cache spec 与 runtime 支持，单 kernel 的护城河下降；仍有价值的是端到端 latent-KV execution、metadata、layout、prefill/decode 切换与 runtime 物化。

## 技术解读主线

正式解读不应从“TokenSpeed 有哪些功能”开始，而应从一个系统问题开始：

> Agentic MoE serving 的瓶颈不只是算子速度，而是请求生命周期、KV page ownership、短 decode 调度间隙、DP cache skew 和 MoE 通信边界能否被一个系统性 runtime 管住。

对应的分析主线：

1. **系统架构：TokenSpeed 是五个运行时面的闭环**  
   画出 Main Python / DP Controller / Scheduler Worker / C++ Scheduler / GPU execution plane 的边界。

2. **请求流：一条消息如何在 TokenSpeed 中执行**  
   展示 admission、ExecutionPlan、CacheOp、ForwardOp、OutputProcessor、scheduler.advance 的闭环。

3. **Agent Runtime / KV Ownership：为什么它可能是最大护城河**  
   用 request FSM 状态表讲清楚 page ownership 如何随 Submitted / Prefilling / Decoding / Draining / Retracting / Finished 迁移。

4. **并行策略：不是支持 EP，而是不同 layer family 的 execution domain**  
   解释 attention DP、dense TP、MoE EP/TP、token-aware comm、RSAG/AR 切换和 placement compiler 的关系。

5. **性能账本：哪些 counter 必须移动**  
   逐项给出 prefix hit、page pressure、GPU gap、collective bytes、rank token skew、p95 ITL/TTFT 等验证指标。

## 源码快照说明

最初深挖主要基于本地 TokenSpeed `ce376da`。本 PR 分支随后已与 `main` 同步到 `cffc875`。本次修订重新复核了当前分支上与 DeepSeek V4 KVStore、parallel mapping、EventLoop、scheduler/KV ownership 相关的关键入口；涉及后续新增代码的细节仍以 `06-code-map-and-open-questions.md` 中的开放问题为准。
