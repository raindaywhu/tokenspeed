# TokenSpeed 架构竞争力与性能归因 v0.3

更新时间：2026-05-24

本篇不写 benchmark 执行计划。它服务于一个更前置的问题：

```text
TokenSpeed 的架构设计是否真的有竞争力？
这种竞争力来自哪些系统机制？
vLLM / vLLM-Ascend 是容易复制，还是需要架构级重构？
```

因此，本篇的重点不是“怎么测”，而是把前面几章的源码分析转成架构判断：

- 这个机制解决了 agentic MoE serving 的什么系统问题？
- 它是否真的区别于通用 serving 后端？
- 如果 vLLM 要复制，是配置、模型 patch、gateway runtime，还是 engine architecture？
- 它可能改善哪类性能瓶颈，为什么不能线性相加？

## 1. 判断前提：目标不是普通 serving

TokenSpeed 是否有竞争力，不能放在普通单轮吞吐场景里判断。它瞄准的更像 V4/Kimi 类 MoE + agentic workload：

| Workload 特征 | 系统压力 |
|---|---|
| 长 system prompt / tool schema / history prefix | repeated prefix 大，TTFT 受 prefix reuse 影响 |
| 多轮 agent session | finish 后生成内容能否安全进入后续 prefix |
| 短 decode step | CPU scheduler / output commit / collective launch 被放大 |
| tool call / structured output / abort | request churn 高，finish/abort 资源释放必须及时 |
| MoE + EP / TP | token dispatch、all-to-all、expert balance 成为瓶颈 |
| latent-KV / compressed attention | KV layout、metadata、cache slot 映射影响 decode |
| 多 DP replica | session KV footprint 和 rank token count 容易 skew |
| p95/p99 敏感 | 平均 tokens/s 不足以代表体验 |

所以，TokenSpeed 的竞争力不能只问 “某个 kernel 快不快”，而要问它是否把这些压力点纳入同一个 runtime 设计。

## 2. 架构竞争力总览

当前源码分析后，可以把 TokenSpeed + SMG 的竞争力拆成五层：

| 架构层 | TokenSpeed / SMG 的实现特征 | 解决的问题 | 竞争力判断 |
|---|---|---|---|
| SMG gateway runtime | OpenAI protocol、chat template、tokenizer cache、tool/reasoning parser、MCP tool loop、worker routing | 让 agent 请求稳定进入 backend，并保持 prompt / routing / parser 一致 | 中等；单点容易复制，完整 gateway runtime 复制成本较高 |
| Scheduler / KV ownership | C++ request FSM、OwnedPages、prefix tree、finish insert、retract/recovery | 多轮 prefix reuse、KV page pressure、finish/abort 生命周期 | 强；属于 runtime protocol，不是配置项 |
| EventLoop / execution loop | cache op、forward op、output feedback 分离并 overlap | 短 decode step 下减少 CPU/GPU 间隙 | 强；需要改 engine 主循环 |
| Split parallelism / model-runtime | attention/dense/MoE 三套 Mapping、process group、CommManager、token-aware comm | 不同 layer family 的最优并行形态不同 | 强；不是“支持 EP”，而是多 group execution protocol |
| local-SPMD / placement compiler | placement annotation、静态插入 collective | 把手写通信规则系统化，降低多模型适配成本 | 中到强；DeepSeek V4 当前不是主路径，但工程价值大 |

这五层不是彼此独立的功能清单。真正的差异在组合：

```text
SMG 保持 prompt/routing 稳定
  -> Scheduler/KV 才有机会复用 prefix
  -> EventLoop 降低短 decode 控制面间隙
  -> split parallelism 减少不合适的 communication shape
  -> MoE / latent-KV backend 才能把模型特性转成吞吐或尾延迟收益
```

如果只复制其中一个点，收益会显著下降。

## 3. SMG：竞争力不在 parser，而在 gateway 一致性

SMG 的 tool parser、reasoning parser 本身不是强护城河。单个 parser 可以被 vLLM 或任意 gateway 复制。

SMG 的竞争力更准确地说是：它把 agent-facing 请求在进入 engine 前统一处理掉。

它做了几件关键事情：

- chat template 和 tokenizer 在 gateway 侧统一执行；
- tokenizer L0/L1 cache 降低 repeated prompt 的 CPU 准备成本；
- tool schema / tool_choice 被转换成 prompt 和 sampling constraint；
- response 被解析成 OpenAI-compatible `tool_calls` / `reasoning_content`；
- Responses API / MCP hosted tools 可以在 gateway 内部循环；
- worker routing 可以在 backend admission 前决定请求落在哪个 worker。

这解决的 workload 问题是：

```text
agent 请求不是裸 prompt，而是 messages + tools + history + tool result + parser state。
如果 gateway 层不稳定，后端 KV cache 和 prefix reuse 很难稳定命中。
```

对 vLLM 的复制判断：

- parser：model/gateway patch，可复制；
- tokenizer cache：gateway runtime，可复制但需要系统化实现；
- MCP tool loop：gateway runtime，可复制但要处理状态、持久化、parser 和下一轮 request 构造；
- routing locality：router + backend 协议，需要知道 worker cache locality，复制难度上升。

因此 SMG 不是最大护城河，但它是 TokenSpeed agent runtime 的入口条件。没有它，engine 的 KV reuse 和多轮稳定性收益会被削弱。

## 4. Scheduler / KV ownership：最强护城河候选

TokenSpeed 的 Scheduler/KV 竞争力不应简化为“有 prefix cache”。真正重要的是：KV page ownership 被纳入 request FSM。

从源码路径看，C++ Scheduler 不是只决定下一批跑多少 token，而是同时管理：

- request lifecycle；
- prefix match；
- active / cached page ownership；
- request-local tail page；
- decode reserve；
- finish 后插入 prefix tree；
- retract / recovery；
- abort / finish 资源释放；
- scheduler.advance 之后的 page 状态反馈。

这解决的是 agentic workload 的核心问题：

```text
多轮请求频繁 finish / resume / abort，KV 既要能复用，又不能被错误复用。
安全复用比“缓存命中”这个口号更难。
```

为什么它可能有竞争力：

1. **生命周期语义深**
   KV page 不是普通 block cache，而是随着 request FSM 在 submitted / prefilling / decoding / draining / finished / retracting 状态间转移。

2. **收益主要体现在尾部稳定性**
   request-local tail page、reserve、retract/recovery 不一定显著提高平均吞吐，但会减少高并发短 decode 下的 page stall 和 queue 抖动。

3. **复制需要改 scheduler 内核**
   vLLM 可以有 prefix cache，但要复制这一整套 ownership 协议，不能只加配置或外部插件。

性能影响的合理归因是：

- repeated prefix 多时，`TTFT` 应下降，因为 prefill tokens saved 增加；
- KV 接近满载时，p95/p99 queue stall 应下降，因为 page ownership 和 recovery 更细；
- request churn 高时，abort/finish 释放更及时，减少无效占用；
- 短 decode 高并发时，tail page waste 和 reserve waste 降低，p95 ITL 更稳定。

需要注意的边界：

- 当前没有看到 tool-call-specific DDR KV offload；
- DeepSeek V4 hierarchical KVStore 短期不能算作已落地收益；
- tool loop 的下一轮 request 是 SMG 发起，不是 engine 内部 pause/resume。

## 5. EventLoop：短 decode 场景的系统收益

Agent workload 常见模式是：

```text
long prefill once
  -> many short decode steps
  -> tool call
  -> next short response
```

短 decode step 下，每步 token 很少，kernel 本身未必是唯一瓶颈。CPU scheduler、output D2H、postprocess、下一轮 admission 和 distributed metadata 同步都可能变成 p95 ITL 的来源。

TokenSpeed 的 EventLoop 竞争力在于它把几个动作拆开：

- scheduler 产出 ExecutionPlan；
- MemoryExecutor 执行 cache op；
- ModelExecutor 执行 forward op；
- OutputProcessor 消费上一轮 output；
- scheduler.advance 用 output 反馈推进 request FSM；
- DP 场景下 forward 前同步各 rank token / batch / mode metadata；
- idle rank 也能参与必要 collective，保证 distributed progress。

这解决的是：

```text
decode 每步太短，控制面开销无法被大 batch 计算掩盖。
```

对 vLLM 的复制判断：

- 简单 async output processor 可以复制；
- 真正复制 `cache op / forward op / output feedback / scheduler.advance` 的闭环，需要 engine 主循环重构；
- DP idle forward、token-aware collective 和 CUDA graph shape 还要和模型执行层协同。

性能上，它更可能改善：

- p95 ITL；
- GPU idle gap；
- scheduler iteration overhead；
- output commit latency；
- 高并发短 decode 下的尾部抖动。

这也是为什么单看平均 throughput 可能低估它的价值。

## 6. 并行策略：不是支持 EP，而是多 execution domain

并行策略是当前最需要讲清楚的部分。TokenSpeed 的特别之处不是“支持 EP/TP/DP”，而是它把不同 layer family 的并行域拆开：

```text
Mapping
  attention: tp / cp / dp
  dense:     tp / dp
  moe:       tp / ep / dp
```

然后把这套 mapping 落到：

- process group；
- weight sharding；
- attention head / KV layout；
- dense TP linear；
- MoE expert placement；
- `CommManager` 的 AR / all-gather / reduce-scatter 选择；
- `ForwardContext.global_num_tokens` 的 token-aware communication；
- MoE backend / all-to-all / grouped expert GEMM。

DeepSeek V4 推荐配置是最有说明力的例子：

```text
--data-parallel-size 4
--enable-expert-parallel
--moe-backend mega_moe
```

按源码推导，实际 execution plan 更接近：

```text
attention: tp=1, dp=4
dense:     tp=4
moe:       ep=4, tp=1
```

这说明 TokenSpeed 不认为 V4 attention、dense、MoE 应该共用同一种 TP。

为什么这可能有竞争力：

1. **attention 和 MoE 的瓶颈不同**
   attention / latent-KV decode 小步更怕 head 切碎和通信延迟；MoE 更怕 expert 权重容量、token dispatch 和 all-to-all。

2. **dense/shared expert 和 routed expert 的策略不同**
   dense/shared expert 可用 TP 摊 GEMM；routed expert 更适合 EP 和专用 dispatch-combine。

3. **短 decode 下 token skew 会放大通信浪费**
   `CommManager` 用 token all-gather / reduce-scatter，而不是简单假设每 rank token 一样多。

4. **隐藏状态 placement 被追踪**
   attention 输出、MLP 输入、MoE 输出之间何时 replicate、shard、partial，不是靠模型作者临时拼通信。

对 vLLM 的复制判断：

- 增加 `--attn-tp-size` / `--moe-tp-size` 风格参数：config-level，容易；
- 让不同 layer family 使用不同 process group：model-runtime，难度中等；
- weight loader、attention layout、MoE backend 都读同一套 mapping：model-runtime，难度较高；
- token-aware AR/RSAG、residual placement、idle rank progress、CUDA graph shape 协同：architecture-level，难；
- Ascend 上还要重建 DeepEP / mega_moe 等价的 all-to-all 或 dispatch-combine backend，难度更高。

边界也要写清楚：

- TokenSpeed 当前禁止 MoE TP 和 EP 同时大于 1；
- CP 目前不是成熟独立策略面；
- DeepSeek V4 不是 generic placement compiler 主路径，而是手写模型逻辑 + `CommManager` + 自定义 V4 attention/MoE。

## 7. local-SPMD / placement compiler：工程竞争力大于短期性能

local-SPMD / placement compiler 不应该被包装成 DeepSeek V4 当前性能收益的主来源。它的价值在另一层：

```text
并行策略是“怎么切”；
placement compiler 是“如何把切分安全落到模型图里”。
```

它用轻量 placement 类型表达：

- `Replicate`
- `Shard`
- `Partial`
- `ATTN_TP`
- `DENSE_TP`
- `MOE_TP_EP`

然后静态分析 decoder layer 的 module spec，插入：

- all-gather；
- reduce-scatter；
- all-reduce；
- deferred reduce；
- residual slice / residual gather；
- fused reduce-norm。

这解决的是复杂模型适配成本问题：

```text
每新增一个模型，如果都手写 CommManager 规则，很容易出错；
placement compiler 可以把通信插入变成可复用协议。
```

对竞争力的判断：

- 直接性能收益不应高估；
- 对 DeepSeek V4 当前主路径的归因要降权；
- 对未来多模型、多并行策略、多硬件适配，工程价值较强；
- vLLM 复制这个能力需要把 model graph、parallel mapping 和 collective insertion 放到同一个抽象里。

## 8. MLA / latent-KV：不是最大护城河

MLA / latent-KV execution 仍有价值，但不应成为报告主线。

原因：

- vLLM 已经吸收或支持 DeepSeek/MLA 相关算子路径；
- 单个 kernel 很容易从护城河变成共享能力；
- Blackwell/CUDA kernel 对 Ascend 不可直接迁移；
- 对 910C/950DT 更有价值的是 layout、metadata、prefill/decode 切换、KV pack/quant、ragged prefill 的系统经验。

合理定位是：

```text
MLA / latent-KV 是模型执行层的加分项；
TokenSpeed 真正值得研究的是 runtime 是否能端到端承载 latent-KV execution。
```

如果收益只来自一个 kernel，那么更合理的路线是反哺 vLLM-Ascend；如果收益来自 Scheduler/KV + split parallelism + latent-KV metadata 的组合，才支持 TokenSpeed 适配价值。

## 9. 竞争力矩阵

| 能力 | 是否独特 | vLLM 复制难度 | 性能影响 | 当前判断 |
|---|---|---|---|---|
| SMG parser | 低 | 低 | correctness / tool compatibility | 不构成核心护城河 |
| SMG gateway runtime | 中 | 中 | TTFT / agent loop latency / routing locality | 有入口价值 |
| Scheduler/KV ownership | 高 | 高 | TTFT、p95 queue、p95 ITL 稳定性 | 核心护城河候选 |
| EventLoop overlap | 高 | 高 | p95 ITL、GPU idle gap | 核心护城河候选 |
| split Mapping | 中高 | 中高 | TPM/device、communication overhead | 核心护城河候选 |
| CommManager token-aware comm | 高 | 高 | collective bytes、rank skew、p95 ITL | 核心护城河候选 |
| placement compiler | 中 | 中高 | 直接性能有限，工程收益强 | 长期护城河候选 |
| DeepSeek V4 mega_moe | 中 | 高但硬件绑定 | MoE throughput | Ascend 可迁移性受限 |
| MLA kernel | 中低 | 中低 | decode throughput | 不应作为主护城河 |

当前最强的三个竞争力候选是：

1. Scheduler / KV ownership；
2. split parallelism + token-aware communication；
3. EventLoop 对短 decode 的执行闭环。

SMG 是必要入口层，但不是最大护城河。placement compiler 是长期工程护城河候选。MLA/kernel execution 比重应降低。

## 10. 性能收益如何写才不空

报告里仍然需要讲性能影响，但写法应该是“机制改变瓶颈”，不是“预计提升 X%”。

| 机制 | 改变的瓶颈 | 可能改善的指标 | 为什么 |
|---|---|---|---|
| SMG tokenizer/cache | gateway prompt preparation | TTFT | repeated template/tokenization 不再每轮重做 |
| SMG routing locality | worker cache locality / load skew | TTFT、queue p95 | 多轮 session 更可能命中同 worker cache |
| Scheduler prefix reuse | repeated prefill compute | TTFT、prefill load | cached prefix 减少重复 prefill |
| finish insert | 后续轮次可复用范围 | TTFT | 生成内容安全进入 prefix tree |
| request-local tail page | KV page fragmentation | p95 queue、p95 ITL | 减少短 decode 尾页浪费 |
| decode reserve | KV admission 稳定性 | p95 ITL | 避免 variable accepted tokens 造成 page 抖动 |
| retract/recovery | KV 满载时的恢复成本 | p95/p99 queue | 用更细粒度恢复替代粗暴失败 |
| EventLoop overlap | CPU/GPU 控制面间隙 | p95 ITL | output feedback 与下一轮 forward 重叠 |
| split parallelism | 不合适的统一 TP/EP | TPM/device、comm time | attention/dense/MoE 分别选择 group |
| token-aware RSAG | rank token skew / padding waste | p95 ITL、comm time | 按真实 token count 通信 |
| MoE EP backend | expert capacity / dispatch-combine | TPM/device、MoE layer time | 专家权重和 token movement 更贴近 MoE 瓶颈 |
| placement compiler | 复杂策略落地成本 | 工程迭代速度、bug rate | 静态插入通信，减少手写错误 |

更谨慎的结论是：

```text
TokenSpeed 的性能竞争力不是单项优化叠加，
而是当 workload 同时具备 long-prefix、short-decode、MoE token skew、
多轮 agent request 时，多个 runtime 协议共同减少无效 prefill、KV stall、
CPU/GPU gap 和不必要通信。
```

## 11. 最终架构判断

如果只看“是否支持 EP、是否有 MLA kernel、是否有 tool parser”，TokenSpeed 的差异并不大。

如果从完整 agentic MoE serving runtime 看，TokenSpeed 的竞争力更明确：

```text
它把 gateway 一致性、request FSM、KV ownership、EventLoop overlap、
layer-family split parallelism、token-aware communication 和模型后端
放进同一套执行协议里。
```

这套系统对 vLLM/vLLM-Ascend 的复制难点不在单点功能，而在协议耦合：

- gateway routing 要服务 backend KV locality；
- scheduler 要知道 page ownership 和 finish/retract 语义；
- model forward 要知道 attention/dense/MoE 的不同 group；
- collective 要知道真实 token distribution；
- output feedback 要反向推进 scheduler；
- MoE backend 要和 mapping/top-k/expert weight layout 一致。

因此当前结论是：

1. TokenSpeed 具备架构级竞争力候选，尤其在 Scheduler/KV、EventLoop、split parallelism 三条主线上。
2. SMG 应作为 agent runtime 入口层单独讲，但不要把 parser 夸大为护城河。
3. local-SPMD / placement compiler 是支撑并行策略长期扩展的工程机制，不是 DeepSeek V4 当前主性能来源。
4. MLA / latent-KV execution 比重应较小，重点介绍实现和迁移边界。
5. 报告的主线应是：TokenSpeed 是否形成了一个更适合 agentic MoE serving 的 runtime，而不是是否有若干 vLLM 也能加的功能点。
