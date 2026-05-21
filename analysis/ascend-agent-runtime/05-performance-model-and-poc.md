# TokenSpeed 性能收益模型与 PoC 胜负线

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

## 4. 四类收益的拆分

### 4.1 减少无效计算

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

### 4.2 降低 KV page pressure

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

### 4.3 减少 CPU/GPU 间隙

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

### 4.4 改变通信账本

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

## 5. 当前收益区间的谨慎估计

以下是 PoC 前假设，不是已经验证的结果：

| 方向 | 可能收益 | 置信度 | 解释 |
|---|---:|---:|---|
| Scheduler / KV ownership / EventLoop | p95/p99 10-25% | 中 | 代码证据强，但依赖 repeated prefix、short decode、KV pressure |
| Agent KV tail / reserve / abort cleanup | p95 稳定性 2-8% | 中 | 更像减少尾部抖动，不一定提升平均吞吐 |
| DP cache-aware dispatch | p95 queue 5-15% | 中低 | 需要多 DP 长会话 trace 验证 |
| split parallelism / token-aware comm | TPM/device 10-30% | 中低 | 取决于 vLLM-Ascend baseline 是否已有等价策略 |
| placement compiler | 直接性能 5-15%，工程收益更大 | 低中 | 只有目标模型实际走 compiler 或需要大量策略试错时才明显 |
| MLA / latent-KV execution | decode 10-30% | 低于护城河 | vLLM 已支持相关算子，单 kernel 护城河下降 |

## 6. 不能提前宣称的收益

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

## 7. 910C + 950DT PoC 胜负线

建议把 PoC 分成两层：

### 7.1 910C：验证能否落地

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

### 7.2 950DT：验证上限

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

## 8. 实验设计建议

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

## 9. 决策规则

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
