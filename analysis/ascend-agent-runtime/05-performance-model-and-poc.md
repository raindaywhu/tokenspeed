# TokenSpeed 性能收益模型与 PoC 胜负线 v0.2

更新时间：2026-05-24

本篇把前面几章的机制分析转成性能账本。重点不是给一个“总收益百分比”，而是回答：

```text
每个机制到底应该移动哪个 counter？
它在什么 workload 下才有效？
vLLM / vLLM-Ascend 如果复制，需要改配置、改模型，还是改 runtime 架构？
```

## 1. 不能简单相加

TokenSpeed 相关优化不能写成：

```text
Scheduler/KV 20% + 并行策略 20% + local-SPMD 10% + MLA 20% = 70%
```

这是错误模型。推理服务每一段时间只有一个或少数几个 active bottleneck：

- 如果 workload 的主要瓶颈是 repeated prefix prefill，prefix reuse 和 cache hit 才会显著影响 TTFT。
- 如果主要瓶颈是短 decode step 的 CPU/GPU 间隙，EventLoop overlap 和 output commit 才会影响 ITL。
- 如果主要瓶颈是 MoE all-to-all / token dispatch，parallel strategy 和 MoE backend 才会影响 TPM/device。
- 如果主要瓶颈是 latent-KV bandwidth，MLA / compressed KV layout 才会影响 decode throughput。
- 如果 baseline 已经有同等能力，TokenSpeed 机制的增量收益会下降。

因此收益模型应该按瓶颈迁移来写：

```text
机制 -> 移动 counter -> 改变瓶颈 -> 影响用户指标
```

## 2. 目标 workload

PoC 目标不是普通单轮吞吐 benchmark，而是更接近 V4/Kimi agent 场景：

| workload 特征 | 对系统的压力 |
|---|---|
| 长 system prompt / tool schema / history prefix | repeated prefix 大，TTFT 受 prefill reuse 影响 |
| 多轮 agent session | finish 后 KV 是否能安全进入下一轮 prefix reuse |
| 短 decode step | CPU scheduler / output processing gap 被放大 |
| tool call / structured output / abort | request churn 高，取消和 finish 资源释放必须及时 |
| MoE + EP / TP | token dispatch、all-to-all、rank skew 成为关键瓶颈 |
| attention latent-KV / compressed KV | decode 侧 KV layout、metadata、cache slot 映射影响 kernel效率 |
| DP serving | 不同 DP rank 的 session KV footprint 可能不均 |
| p95/p99 latency 敏感 | 平均 tokens/s 不能代表真实体验 |

## 3. 机制到 counter 的映射

| 机制 | 应该移动的 counter | 影响指标 | 有效前提 |
|---|---|---|---|
| SMG tokenizer/cache | gateway prepare latency、tokenizer L0/L1 hit rate | TTFT | 多轮 prompt/template 稳定 |
| SMG worker routing | routing affinity、worker load skew、backend prefix hit | TTFT、p95 queue | 多 worker / 多 DP，session 有重复 prefix |
| SMG tool loop | tool loop iteration latency、parser latency、tool-call success rate | E2E latency、structured output 成功率 | Responses/MCP hosted tools 占比高 |
| prefix match / finish insert | cached tokens、prefill tokens saved、fully cached prefill count | TTFT、TPM/device | 多轮 repeated prefix 占比高 |
| request-local tail page | tail page waste、active KV pages、page admission failure | p95 ITL、queue stall | 短 decode / 高并发导致 page pressure |
| dynamic decode reserve | reserved tokens / accepted tokens ratio、reserve update count | p95 ITL、device KV stability | spec decode 或 variable accept length 明显 |
| retract / recovery | retraction count、writeback/loadback latency、recompute avoided | p95/p99 queue delay、KV pressure | device KV 接近满载 |
| cache op / forward op 分离 | loadback stall、writeback stall、cache op overlap | TTFT、ITL | hierarchical cache 可用且 transfer 能 overlap |
| EventLoop overlap | GPU idle gap、scheduler iteration time、CPU commit time | p95 ITL | decode step 短，CPU postprocess 可见 |
| DP cache-aware dispatch | pages/rank variance、queue/rank variance、retraction/rank | p95 queue time | 多 DP replica 且 session KV skew |
| token-aware comm | collective bytes、AG/RS/AR time、padding waste | TPM/device、TPS/user | rank 间 token 数不均或 layer group 不同 |
| split parallelism | attention/dense/MoE 各段 kernel time、comm time | TPM/device | 最优 attention/dense/MoE parallel shape 不同 |
| placement compiler | extra collective count、strategy iteration cost、bug rate | 间接收益 | 多模型/多策略需要快速试错 |
| MLA / latent-KV | KV bytes/token、decode attention time、metadata build time | decode throughput | latent-KV bandwidth / metadata 是瓶颈 |

## 4. 归因模型：先分层，再看瓶颈迁移

PoC 里最容易犯的错误是看到 `TTFT` 或 `TPM/device` 变好，就把收益归给某个最显眼的模块。更稳的做法是把一次请求拆成几段：

```text
T_e2e =
  T_gateway_prepare
+ T_gateway_routing
+ T_queue
+ T_prefill_uncached
+ T_prefill_cached_overhead
+ T_decode_compute
+ T_decode_comm
+ T_output_commit
+ T_tool_loop_external_or_mcp
```

这个式子不是要求精确相加，而是给性能归因定边界：

- `T_gateway_prepare` 下降，通常是 SMG tokenizer/template cache 或 parser 准备阶段收益。
- `T_queue` 下降，可能来自 Scheduler admission、更少 page pressure，也可能来自 SMG routing 更均衡。
- `T_prefill_uncached` 下降，通常要看 cached token ratio / prefill tokens saved，而不是只看 TTFT。
- `T_decode_compute` 下降，可能来自 attention/MLA kernel、MoE backend、CUDA graph 或更好的 batch shape。
- `T_decode_comm` 下降，才更能证明 split parallelism / token-aware collective 的价值。
- `T_tool_loop_external_or_mcp` 下降，归因在 gateway / agent runtime，不应包装成 engine KV 优化。

### 4.1 判断收益归属的最小证据链

| 观察到的变化 | 必须同时看到的 counter | 才能归因给 |
|---|---|---|
| TTFT 下降 | gateway prepare 不变，prefill tokens saved 上升 | engine prefix reuse / KV ownership |
| TTFT 下降 | gateway prepare 下降，prefill tokens saved 不变 | SMG tokenizer/cache/template |
| p95 ITL 下降 | GPU idle gap / output commit time 下降 | EventLoop overlap / output feedback |
| p95 ITL 下降 | collective duration / bytes 下降 | split parallelism / token-aware comm |
| TPM/device 上升 | MoE dispatch/combine time 下降 | MoE backend / EP dispatch-combine |
| TPM/device 上升 | decode attention time 下降，KV bytes/token 下降 | latent-KV / attention kernel |
| queue time 下降 | page pressure / retraction / admission failure 下降 | Scheduler/KV ownership |
| queue time 下降 | worker load skew 下降，但 page pressure 不变 | SMG routing / DP load balancing |

如果只看到 end-to-end latency 变化，无法做护城河判断。护城河判断必须落到“哪个 runtime 协议复制成本高”。

## 5. 并行策略收益的拆解实验

并行策略尤其不能只做 “TokenSpeed vs vLLM-Ascend” 的黑盒对比。需要把 execution plan 拆成几组，否则无法判断收益来自 layer-family split、EP backend、还是某个 kernel。

建议至少做四个 plan：

| Plan | Attention | Dense/shared | Routed MoE | 目的 |
|---|---|---|---|---|
| A uniform TP baseline | TP=N | TP=N | TP 或 baseline EP | 模拟单一并行组 |
| B split no EP | 小 TP / DP | TP=N | MoE TP | 看 attention/dense/MoE 拆组本身 |
| C split + EP | 小 TP / DP | TP=N | EP | 看 V4/Kimi 核心 execution plan |
| D split + EP + optimized dispatch | 同 C | 同 C | EP + low-latency all-to-all / grouped GEMM | 看 MoE backend 上限 |

每组必须采集：

- attention forward time，特别是 decode 下 small-step 的 per-token time；
- dense/shared expert GEMM time；
- `pre_mlp_comm` / `post_mlp_comm` / all-gather / reduce-scatter time；
- MoE router、dispatch、expert compute、combine time；
- all-to-all bytes / duration / rank token skew；
- attention/dense/MoE 各自 group 的 local token count；
- p50/p95 ITL、TPM/device、TPS/user。

### 5.1 为什么这能回答“并行策略特别在哪里”

如果 A -> B 有收益，说明 **layer-family split** 有价值：attention 不应该被 dense/MoE 的 TP 形态绑架。

如果 B -> C 有收益，说明 **EP 解决了 routed expert 的容量或计算瓶颈**。

如果 C -> D 有收益，说明 **EP 本身不够，dispatch-combine backend 才是性能关键**。

如果 A/B/C/D 的 p95 ITL 差异很小，但平均 TPM 变化大，说明收益更偏吞吐，不一定解决 agent short-decode latency。

如果 C/D 的 TPM 上升但 p95 ITL 变差，说明 MoE all-to-all 或 rank skew 在尾延迟上吃掉了吞吐收益。

## 6. vLLM / vLLM-Ascend 复制难度判定矩阵

报告里不要只问 “vLLM 能不能做”。要按复制层级拆：

| TokenSpeed/SMG 能力 | 复制层级 | 复制难点 | 护城河判断 |
|---|---|---|---|
| tool-call parser | model/gateway patch | 模型格式覆盖、streaming delta | 弱到中 |
| SMG tokenizer/cache | gateway runtime | L0/L1 cache、template 一致性、prefix 稳定 | 中 |
| SMG worker routing | router/runtime | worker affinity、KV locality、load skew tradeoff | 中到强 |
| Scheduler request FSM | runtime protocol | request 状态与 page ownership 绑定 | 强 |
| prefix tree / finish insert | runtime protocol | finished output 安全进入可复用 prefix | 强 |
| request-local tail page / reserve | scheduler + allocator | page 生命周期、retract/recovery 正确性 | 强 |
| EventLoop overlap | runtime execution loop | cache op / forward op / output feedback 解耦 | 强 |
| split Mapping | model-runtime | attention/dense/MoE 多套 group | 中到强 |
| CommManager token-aware comm | model-runtime + collective | AR/RSAG 切换、residual placement、idle forward | 强 |
| placement compiler | model compiler | placement 类型系统、静态通信插入 | 中，工程价值强 |
| DeepSeek V4 mega_moe | backend/kernel | CUDA/DeepGEMM/SM100 强绑定 | 对 Ascend 可迁移性低 |
| MLA kernel | operator/backend | vLLM 已吸收相关路径，单 kernel 稀缺度下降 | 弱到中 |

这张表的含义是：如果 PoC 收益主要来自 parser、单 kernel 或简单 flag，TokenSpeed 作为整体适配对象的必要性下降；如果收益来自 Scheduler/KV + split execution protocol + routing locality 的组合，才说明 vLLM/vLLM-Ascend 复制成本高。

## 7. PoC 数据采集协议

为了让结果能支持决策，PoC 需要从第一天就按层采集数据。

### 7.1 Gateway / SMG 层

需要采集：

- request prepare latency；
- tokenizer L0/L1 hit rate；
- chat template rendering time；
- worker selection policy、selected worker、routing key；
- tool parser latency；
- MCP tool loop iteration count / latency；
- structured output parse success / failure。

目的：避免把 gateway 改善误归因给 engine。

### 7.2 Scheduler / KV 层

需要采集：

- waiting/running request 数；
- queue time；
- active/cached/free pages；
- prefix matched tokens；
- prefill tokens saved；
- request-local tail waste；
- decode reserve vs accepted tokens；
- finish insert count；
- retract/writeback/loadback count 和 latency；
- abort 后资源释放延迟。

目的：证明 agent 多轮收益是否来自 KV ownership，而不是普通吞吐波动。

### 7.3 Execution / EventLoop 层

需要采集：

- scheduler iteration duration；
- cache op duration；
- forward op duration；
- output D2H copy wait；
- output commit/postprocess time；
- GPU idle gap；
- CUDA graph replay hit/miss；
- idle forward count。

目的：判断短 decode step 下 CPU/GPU 间隙是否被压缩。

### 7.4 Parallel / Communication 层

需要采集：

- attention/dense/MoE group 配置；
- 每层 all-reduce / all-gather / reduce-scatter / all-to-all 次数；
- collective bytes 和 duration；
- rank token count variance；
- padding waste；
- MoE expert token balance；
- dispatch/combine duration；
- fused reduce+norm 命中率。

目的：证明并行策略收益来自通信账本变化，而不是 workload 偶然差异。

### 7.5 Model / Kernel 层

需要采集：

- attention prefill/decode/mixed backend time；
- KV bytes/token；
- metadata build time；
- MoE router / grouped GEMM / shared expert time；
- latent-KV pack/quant/dequant time；
- MLA / sparse attention fallback rate。

目的：分离 kernel 收益和 runtime 收益。

## 8. 四类收益的拆分

### 8.1 减少无效计算

对应机制：

- prefix cache hit；
- fully cached prefill；
- finish 后把生成结果转成后续可复用 prefix；
- grammar admission 避免未 ready 请求占 prefill slot；
- abort 及时释放 request slot。

应该看：

- prefill input tokens 是否下降；
- cached token ratio 是否上升；
- 已取消请求是否继续占 active slot；
- queue time 是否下降；
- TTFT 是否改善。

这类收益最可能体现在 repeated-prefix agent traces。没有重复 prefix 的单轮吞吐 benchmark 可能看不出。

### 8.2 降低 KV page pressure

对应机制：

- `OwnedPages` / request FSM ownership；
- request-local tail page；
- dynamic decode reserve；
- retraction；
- paged cache group。

应该看：

- device free pages；
- active vs cached pages；
- tail page available tokens；
- reserve tokens 与实际 accepted tokens 的差距；
- retraction count；
- page allocation failure；
- req pool slot occupancy。

这类收益通常不是 p50 tokens/s，而是 p95/p99 稳定性：高并发短 decode 下少 stall、少全局阻塞。

### 8.3 减少 CPU/GPU 间隙

对应机制：

- `event_loop_overlap()`；
- output D2H copy event；
- current forward 与 previous result commit 错位；
- idle forward 参与 distributed collectives；
- CUDA graph / static buffer。

应该看：

- GPU forward 之间的 idle gap；
- scheduler iteration duration；
- output processor postprocess time；
- sampling + copy D2H time；
- zero-token idle forward count；
- CUDA graph replay hit ratio。

对短 decode step，这类收益可能比 kernel 微优化更重要，因为每步 token 很少，CPU 控制面开销会占更高比例。

### 8.4 改变通信账本

对应机制：

- attention / dense / MoE mapping 分开；
- attention DP + dense TP + MoE EP；
- `CommManager.use_all_reduce()` 决定 AR vs token AG/RS；
- `ForwardContext.global_num_tokens` 携带跨 DP rank token 分布；
- DeepEP / MegaMoE backend；
- placement compiler 自动插入通信。

应该看：

- all-reduce / all-gather / reduce-scatter / all-to-all 次数；
- collective bytes；
- collective duration；
- rank token count variance；
- padding waste；
- expert token balance；
- MoE dispatch/combine time。

这一类是“并行策略实现”能否成为护城河的关键。不能只说支持 EP，而要证明某个策略减少了真实 token movement 或降低了某个 layer family 的瓶颈。

## 9. 当前收益区间的谨慎估计

以下是 PoC 前假设，不是已经验证的结果：

| 方向 | 可能收益 | 置信度 | 解释 |
|---|---:|---:|---|
| Scheduler / KV ownership / EventLoop | p95/p99 10-25% | 中 | 代码证据强，但依赖 repeated prefix、short decode、KV pressure |
| Agent KV tail / reserve / abort cleanup | p95 稳定性 2-8% | 中 | 更像减少尾部抖动，不一定提升平均吞吐 |
| DP cache-aware dispatch | p95 queue 5-15% | 中低 | 需要多 DP 长会话 trace 验证 |
| split parallelism / token-aware comm | TPM/device 10-30% | 中低 | 取决于 vLLM-Ascend baseline 是否已有等价策略 |
| SMG gateway/tokenizer/tool loop | TTFT/E2E 2-10% | 中低 | 取决于 tokenizer cache 命中和 MCP/tool loop 占比 |
| placement compiler | 直接性能 5-15%，工程收益更大 | 低中 | 只有目标模型实际走 compiler 或需要大量策略试错时才明显 |
| MLA / latent-KV execution | decode 10-30% | 低于护城河 | vLLM 已支持相关算子，单 kernel 护城河下降 |

## 10. 不能提前宣称的收益

1. **不能宣称 tool call 后 KV 专门下沉到 DDR。**  
   当前没有看到 tool-call-specific offload policy。tool call 更像正常 finish / stop token 路径。

2. **不能把 DeepSeek V4 hierarchical KVStore 算进短期收益。**  
   当前 DeepSeek V4 baseline 仍要求 `--disable-kvstore`，offload/reload 未实现。

3. **不能把 local-SPMD 说成 DeepSeek V4 主路径。**  
   V4 目前更像 `CommManager + model-specific logic`，placement compiler 是重要基础设施但不是已确认主执行路径。

4. **不能用平均吞吐替代 agent 指标。**  
   目标 workload 关心 p95 TTFT、p95 ITL、queue stall、cache hit、rank skew。

5. **不能只拿 MLA kernel 作为适配理由。**  
   如果收益只来自一个 vLLM 已吸收的 kernel，应优先反哺 vLLM-Ascend，而不是迁移整个 TokenSpeed runtime。

6. **不能把 SMG tool loop 说成 engine 内部 resume。**
   MCP hosted tool loop 是 gateway 发起下一轮 backend request，不是 TokenSpeed engine 内同一个 request pause/resume。

7. **不能把 DP routing 改善直接等同于 KV cache 改善。**
   如果 prefix hit 没上升，只是 worker load 更均衡，那收益属于 routing/load balancing，不属于 prefix KV reuse。

## 11. 910C + 950DT PoC 胜负线

建议把 PoC 分成两层：

### 11.1 910C：验证能否落地

目标：

- 跑通 DeepSeek V4 / Kimi-like MoE 的基本 serving；
- 保持工具调用、stop、streaming、质量不退化；
- 验证 Scheduler / KV ownership 相关 counters 能采集；
- 验证 Ascend/HCCL 下 parallel mapping 是否能表达 attention/dense/MoE 分组；
- 先不把 MLA / latent-KV 算作最大收益来源。

最低胜负线：

- 相比 vLLM-Ascend，`TPS/user` 不退化；
- p95 ITL 不退化；
- repeated-prefix trace 的 TTFT 有可解释改善；
- 没有质量、stop、tool behavior 回归。

### 11.2 950DT：验证上限

目标：

- 在更接近目标硬件上验证 parallel strategy 和 latent-KV execution；
- 观察 MoE all-to-all / token dispatch 是否成为瓶颈；
- 验证 attention/dense/MoE split strategy 是否能降低通信或提高设备利用率；
- 评估 950DT 特有算子路径下，TokenSpeed runtime 的优势是否仍存在。

强胜负线：

- `TPM/device` 提升 20-30%；
- `TPS/user` 不退化；
- p95 TTFT / p95 ITL 不退化，最好改善；
- MoE communication counters 有明确改善，或能解释同等性能下调参更简单；
- 质量与工具调用行为不退化。

## 12. 实验设计建议

至少需要四组 trace：

1. **单轮长上下文**：用于分离 prefill / latent-KV / attention kernel。
2. **多轮 repeated-prefix agent**：用于验证 prefix reuse、finish insert、TTFT。
3. **短 decode 高并发**：用于验证 EventLoop overlap、tail page、reserve、p95 ITL。
4. **MoE rank skew trace**：用于验证 DP cache-aware dispatch、token-aware comm、EP/MoE backend。

每组 trace 都应同时采集：

- user metrics：TTFT、ITL、E2E latency、TPS/user、TPM/device；
- scheduler metrics：active requests、queue time、scheduled tokens、finish/abort/retract；
- KV metrics：device/host pages、cached pages、tail waste、reserve waste、prefix hit；
- runtime metrics：CPU commit time、GPU idle gap、D2H copy wait；
- communication metrics：collective count、bytes、duration、rank token distribution；
- quality metrics：输出一致性、stop/tool behavior、structured output 成功率。

## 13. 决策规则

如果 PoC 发现收益主要来自：

```text
Scheduler/KV ownership + EventLoop overlap + parallel strategy
```

那么 TokenSpeed 值得进入 910C/950DT 适配，因为这说明收益来自 runtime 架构组合，vLLM/vLLM-Ascend 复制成本较高。

如果收益主要来自：

```text
单个 MLA / latent-KV kernel
```

则不应优先迁移整个 TokenSpeed。更合理路线是把算子和 layout 经验反哺 vLLM-Ascend。

如果收益来自：

```text
config-level parallelism knobs
```

则需要进一步判断 vLLM-Ascend 是否能通过现有配置或小范围 model patch 达到同样效果；这类收益不足以构成 TokenSpeed 的核心护城河。
