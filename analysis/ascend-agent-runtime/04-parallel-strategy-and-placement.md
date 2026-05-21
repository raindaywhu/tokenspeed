# TokenSpeed 并行策略 / Placement Compiler 深挖 v0.1

日期：2026-05-18  
源码快照：主体分析基于 TokenSpeed `ce376da`，本 PR 分支同步到 `cffc875` 后已复核 split parallelism、DeepSeek V4 kvstore 限制、`CommManager` / placement compiler 关键入口。  
本篇范围：分析 TokenSpeed 如何把 attention / dense / MoE / DP / EP / TP / CP 这些并行维度落到执行图里，以及 vLLM/vLLM-Ascend 复制它的难度。  
重要边界：本篇不重新深挖 vLLM 当前实现；只基于 TokenSpeed 源码判断“如果 vLLM 要复制，属于 config、model patch，还是 architecture-level 复制”。

## 0. 核心判断

TokenSpeed 的并行策略实现不应该被简化成“多了几个 CLI 参数”。它的实现链路是：

```text
server args
  -> resolve_parallelism()
  -> Mapping(attn / dense / moe)
  -> process groups
  -> model weights / attention layout / MoE backend
  -> CommManager 或 placement compiler 插入通信
  -> ForwardContext 携带 token 分布信息
```

真正有研究价值的是这条链路把并行策略从“启动参数”变成了模型执行协议的一部分。

但这里也要纠偏一个容易过度包装的说法：

- TokenSpeed 确实支持 attention TP、dense TP、MoE TP、MoE EP、attention DP 的拆分表达。
- 但是源码明确禁止 MoE TP 和 EP 同时大于 1。
- CP 目前通过 `ENABLE_CP` 环境开关把 `attn_tp_size` 解释为 CP size，不能把它视为成熟的多维 CP 策略接口。
- local-SPMD / placement compiler 不是所有模型都走的路径。DeepSeek V4 走的是手写模型逻辑 + `CommManager` + 自定义 V4 attention/MoE；`CompiledDecoderLayer` 目前更像是 GPT-OSS 等模型的可复用路径。

所以本篇的核心判断是：

**TokenSpeed 的护城河候选不是“支持 EP”或“有 placement annotation”，而是：它已经把不同 layer family 的并行拓扑、通信边界、token 分布、权重切分、MoE backend 选择整合成一个可执行的模型层协议。**

如果 vLLM/vLLM-Ascend 只是增加 `--attn-tp-size`、`--dense-tp-size`、`--moe-tp-size` 这类参数，并不等价于复制。等价复制需要解决：

1. 每个 layer family 的 rank group 如何定义。
2. 权重如何按不同 group 切分和加载。
3. attention / dense / MoE 之间的 hidden/residual placement 如何转换。
4. all-reduce、all-gather、reduce-scatter、all-to-all 哪些该发生，哪些该省掉。
5. 短 decode step 下，不同 DP rank 的 token 数不一致时，collective 怎么仍然正确且不被 padding 浪费拖垮。
6. MoE EP backend、DeepEP/All2All 和模型层 TopK / shared expert / routed expert 之间如何配合。

这属于 architecture-level 复制，至少是 model-runtime 协同级别复制，不是 config-level 复制。

## 1. 最关键的具体例子：DeepSeek V4 文档里的并行形态

TokenSpeed 的 DeepSeek V4 文档推荐：

```bash
--data-parallel-size 4
--enable-expert-parallel
--moe-backend mega_moe
```

证据：`docs/serving/deepseek-v4.md:1-31`

这条命令表面看是 DP + EP，但按 `resolve_parallelism()` 实际推导：

```text
world_size = data_parallel_size = 4

attention:
  tp = 1
  cp = 1
  dp = 4

dense:
  tp = world_size = 4
  dp = 1

moe:
  ep = world_size = 4
  tp = world_size / ep = 1
  dp = 1
```

证据：

- `python/tokenspeed/runtime/utils/server_args.py:409-475`
- `python/tokenspeed/runtime/distributed/mapping.py:133-232`
- `python/tokenspeed/runtime/distributed/mapping.py:260-347`

这就是我们报告里应该抓住的“并行策略实现”含金量：

**V4 的 recommended execution plan 不是 uniform TP，也不是简单 EP，而是 attention 走 DP replica，dense/共享计算走 TP 或 MoE group，routed experts 走 EP，MoE backend 走 `mega_moe`。**

对 V4/Kimi 类 MoE，这比“支持 EP”本身更重要。因为 attention、dense、MoE 的瓶颈不同：

- attention / latent-KV：decode 小 batch 时更怕 head 切得太碎、KV layout 复杂、通信延迟占比上升。
- dense/shared expert：权重和激活矩阵更适合用 TP 摊计算。
- routed MoE：专家权重容量和 token dispatch 是核心，需要 EP 或专门 all-to-all。
- agentic workload：短 decode step、多轮 prefix、rank 间 token 数不均，会放大 collective launch、padding、idle rank 的代价。

## 2. 证据入口

| 层级 | 关键入口 | 说明 |
| --- | --- | --- |
| CLI / 参数 | `python/tokenspeed/runtime/utils/server_args.py:409-475`, `1568-1610`, `1690-1735` | 解析 `--attn-tp-size`、`--dense-tp-size`、`--moe-tp-size`、`--data-parallel-size`、`--enable-expert-parallel` |
| Mapping | `python/tokenspeed/runtime/distributed/mapping.py:27-347` | 定义 attention/dense/MoE 三套 rank/group |
| Process group | `python/tokenspeed/runtime/execution/distributed_initializer.py:38-163` | 初始化 world、attention TP、dense TP、MoE TP/EP process groups |
| 手写通信层 | `python/tokenspeed/runtime/distributed/comm_manager.py:37-337` | BaseDecoderLayer 默认路径，按 layer family 决定 AR / RSAG |
| placement 类型系统 | `python/tokenspeed/runtime/models/base/placement.py:21-113` | Replicate / Shard / Partial 与 ATTN_TP / DENSE_TP / MOE_TP_EP |
| compiler | `python/tokenspeed/runtime/models/base/compiler.py:182-620` | 根据 ModuleSpec 插入通信 op |
| compiled runtime | `python/tokenspeed/runtime/models/base/execution.py:70-156` | 执行 compiler 生成的 step/pre_comms/post_comms |
| base decoder | `python/tokenspeed/runtime/models/base/decoder_layer.py:68-368` | `BaseDecoderLayer` 使用 `CommManager`，`CompiledDecoderLayer` 使用 compiler |
| base transformer | `python/tokenspeed/runtime/models/base/transformer_model.py:82-220` | 编译 decoder stack，处理 final norm / LM head 前通信 |
| MoE layer | `python/tokenspeed/runtime/layers/moe/layer.py:31-193` | 统一 MoE spec，禁止 TP+EP 同时开启 |
| DeepEP dispatcher | `python/tokenspeed/runtime/layers/moe/dispatcher/deep_ep.py:37-658` | normal / low_latency dispatch-combine |
| DeepSeek V4 | `python/tokenspeed/runtime/models/deepseek_v4.py:3121-3715`, `4785-4895`, `5515-5845` | V4 attention、MoE、decoder layer 的实际并行路径 |
| Qwen3.5 MoE | `python/tokenspeed/runtime/models/qwen3_5_moe.py:70-360` | 更接近 base CommManager 的 MoE 实例 |

## 3. Strategy 表达：不是一个 TP，而是三套 layer family mapping

### 3.1 `_resolve_parallelism_sizes`

`_resolve_parallelism_sizes(world_size, *sizes)` 的规则是：

1. 输入按“innermost 到 outermost”排列。
2. 如果某个 size 是 `None`，用剩余 world size 推断。
3. 最终所有 size 的乘积必须等于 `world_size`。

证据：`python/tokenspeed/runtime/distributed/mapping.py:27-47`

这意味着 rank layout 是一个显式设计，不只是参数校验。

### 3.2 AttentionLayerMapping

Attention mapping 有：

- `tp_size`
- `cp_size`
- `dp_size`
- `tp_group`
- `cp_group`
- `dp_group`

rank stride 是：

```text
attention TP: stride = 1
attention CP: stride = tp_size
attention DP: stride = tp_size * cp_size
```

证据：`python/tokenspeed/runtime/distributed/mapping.py:133-187`

含义：

- attention TP 是最快变化维度。
- DP rank 是外层维度。
- 对 serving 来说，这让一个 DP replica 内可以有自己的 attention TP group，同时多个 DP replica 独立接请求。

### 3.3 DenseLayerMapping

Dense mapping 只有：

- `tp_size`
- `dp_size`

证据：`python/tokenspeed/runtime/distributed/mapping.py:94-132`

默认行为很重要：如果用户没有指定 `--dense-tp-size`，`resolve_parallelism()` 会把 dense TP 设成 `world_size`。

证据：`python/tokenspeed/runtime/utils/server_args.py:445-450`

这导致一种常见策略：

```text
attention: DP replica, 小 TP 或 TP=1
dense: 全 world TP
```

这不是常规统一 TP 的系统能自然表达的。

### 3.4 MoeLayerMapping

MoE mapping 有：

- `tp_size`
- `ep_size`
- `dp_size`
- `tp_ep_size`
- `tp_group`
- `ep_group`
- `tp_ep_group`

证据：`python/tokenspeed/runtime/distributed/mapping.py:189-258`

MoE 运行通信主要使用 `tp_ep_group`，也就是把 MoE TP 和 EP 看成一个通信边界：

```text
moe.tp_ep_size = moe.tp_size * moe.ep_size
moe.tp_ep_group = contiguous ranks with stride=1
```

但是源码有强约束：

```text
if mapping.moe.has_tp and mapping.moe.has_ep:
    raise ValueError("MoE TP and EP cannot be both > 1")
```

证据：`python/tokenspeed/runtime/utils/server_args.py:477-478`

`MoELayer` 也再次禁止：

```text
if tp_size > 1 and ep_size > 1:
    raise ValueError("Mixed TP and EP is not supported yet.")
```

证据：`python/tokenspeed/runtime/layers/moe/layer.py:84-98`

所以报告里应该避免说 TokenSpeed 已经支持任意 TP+EP 组合。更准确的说法是：

**TokenSpeed 支持在 attention/dense/MoE 三个 layer family 之间选择不同并行策略；MoE 内部目前是在 TP-only 和 EP-only 之间选，而不是任意 TPxEP。**

这个限制不削弱研究价值，反而让结论更清楚：它的优势在“为 workload 选择更合适的分解”，不是“并行维度无限组合”。

### 3.5 CP 的边界

`resolve_parallelism()` 里如果 `ENABLE_CP` 为真，会把 `attn_tp_size` 解释成 `attn_cp_size`，并把 `attn_tp_size` 设为 1。

证据：`python/tokenspeed/runtime/utils/server_args.py:418-421`

这说明 CP 还不是和 TP/DP 对等的完整 CLI 策略面。报告可以提 CP，但应该降权：

- 可说：源码有 CP slot，mapping 层能表达 attention CP。
- 不应说：TokenSpeed 已经展示了成熟 CP execution plan。

## 4. Process group：策略变成 runtime 拓扑

`DistributedInitializer.initialize()` 会基于 mapping 初始化：

- `world_group`
- `attn.tp_group`
- `dense.tp_group`
- `moe.tp_ep_group`

证据：`python/tokenspeed/runtime/execution/distributed_initializer.py:152-163`

这个选择说明：

1. Runtime 并不只知道一个 tensor-parallel group。
2. attention、dense、MoE collectives 可以走不同 group。
3. MoE 统一使用 `tp_ep_group`，便于对 TP-only 和 EP-only 共用通信边界。
4. DP 主要通过 outer rank layout、request routing、CPU metadata gather、attention DP rank 控制，而不是把每个 layer 都做 DP collective。

DP 相关路径：

- `DataParallelController` 按 `dp_rank * attn_tp_size + attn_tp_rank` 启动每个 attention TP group。
- 每个 rank 的 mapping 会写入具体 `global_rank`。
- EventLoop 在 forward 前用 CPU all-gather 收集各 DP rank 的 token 数、batch size、forward mode。

证据：

- `python/tokenspeed/runtime/engine/data_parallel_controller.py:276-314`
- `python/tokenspeed/runtime/engine/event_loop.py:858-901`
- `python/tokenspeed/runtime/execution/model_executor.py:880-888`

这对短 decode workload 很关键：不同 DP replica 同一轮可能有不同 token 数，甚至有 rank idle。TokenSpeed 会把 `global_num_tokens` 放进 `ForwardContext`，后续 token-aware collective 才能知道每个 rank 的真实 token 分布。

## 5. CommManager：默认路径如何把策略落到 decoder layer

`BaseDecoderLayer` 默认不走 compiler，而是每层构造一个 `CommManager`。

证据：`python/tokenspeed/runtime/models/base/decoder_layer.py:68-150`

这条路径的核心是：在 attention、MLP 前后显式调用通信函数。

### 5.1 Token 分布不是假设均匀 padding，而是按真实 token count

`CommManager.scattered_num_tokens(ctx)` 会：

- 如果有 `ctx.global_num_tokens`，按 attention DP rank 展开每个 rank 的 token 数。
- 否则按当前 rank 的 `ctx.input_num_tokens` 均分到 attention TP group。

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:37-80`

然后它分别计算：

- attention TP group 的 token count
- dense TP group 的 token count
- MoE TP/EP group 的 token count

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:55-104`

这不是小细节。短 decode step 下，collective 的成本经常不是理论带宽，而是：

- rank token 数不均造成 padding 浪费。
- idle rank 仍要参加 collective。
- small message launch latency 占比变高。

TokenSpeed 的通信 op 不是只看 tensor shape，而是从 scheduler/event loop 传入的 `ForwardContext` 得到真实 token 分布。

### 5.2 AR vs RSAG：取决于相邻 layer family group 是否等价

`CommManager.use_all_reduce(is_moe)` 的规则：

```text
如果下游是 MoE:
  use_all_reduce = attn.tp_size == moe.tp_ep_size
否则:
  use_all_reduce = attn.tp_size == dense.tp_size
```

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:96-104`

含义：

- 如果 attention TP 和 dense TP 一样大，就走 all-reduce 风格，保留“每 rank 有相同 token”的模式。
- 如果 group size 不一样，就走 token all-gather / token reduce-scatter，进入 RSAG 模式。

这对 split parallelism 是核心。否则不同 layer family 的 TP size 不同，只会导致 hidden state shape 对不上。

### 5.3 Attention 前后通信

attention 前：

- 第一层不需要 pre-attn comm。
- 如果上一层输出已经和 attention group 对齐，直接走。
- 如果上一个 layer family 输出是 shard，需要 `token_all_gather`。

attention 后：

- 如果可用 all-reduce，就 all-reduce attention 输出。
- 如果从 RSAG 切回 AR，还要 all-gather residual。
- 如果不可用 all-reduce，就 `token_reduce_scatter`，并在必要时 slice residual。

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:106-163`

这里的关键不是 primitive 本身，而是 hidden 和 residual 同时有 placement 状态。如果 residual 没跟着正确 gather/slice，模型输出会错。

### 5.4 Dense / MoE 前后通信

dense 前：

- 如果 dense TP 不存在，直接执行。
- 如果 attention group 和 dense group 不一致，先 `token_all_gather`。

MoE 前：

- 如果 MoE TP/EP group 不存在，直接执行。
- 如果 attention group 和 MoE group 不一致，先 `token_all_gather`。

dense/MoE 后：

- all-reduce 或 reduce-scatter，规则同上。

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:165-252`

这说明 split parallelism 的实现重点不是“每层支持 TP”，而是“跨 layer family 的通信转换”。这也是 vLLM 复制时最容易低估的部分。

### 5.5 Fused allreduce + norm

`CommManager` 还会在满足条件时把 all-reduce 和 RMSNorm 融合：

- 当前模式可 all-reduce。
- attention TP 存在。
- `enable_allreduce_fusion` 开启。
- token 数不超过 `comm_fusion_max_num_tokens`。

证据：`python/tokenspeed/runtime/distributed/comm_manager.py:254-337`

`server_args.resolve_communication()` 里，如果 attention TP 和 dense TP 不同，会直接禁用 allreduce fusion：

```text
if mapping.attn.tp_size != mapping.dense.tp_size:
    comm_fusion_max_num_tokens = -1
    enable_allreduce_fusion = False
```

证据：`python/tokenspeed/runtime/utils/server_args.py:541-548`

这很重要：split parallelism 不是免费收益。它可能减少不必要的 all-reduce，也可能让 fused allreduce+norm 失效。性能判断必须按瓶颈建模，不能简单说“split 一定更快”。

## 6. Placement compiler：local-SPMD 如何服务并行策略

### 6.1 Placement 类型系统

`placement.py` 定义三个 tensor 状态：

- `Replicate(group)`：每个 rank 都有完整 tensor。
- `Shard(group)`：按 token 维分散在 ranks。
- `Partial(group)`：每个 rank 是部分和，需要 reduce 才完整。

并定义三个 group：

- `ATTN_TP`
- `DENSE_TP`
- `MOE_TP_EP`

证据：`python/tokenspeed/runtime/models/base/placement.py:21-113`

这个抽象非常克制，不是完整 DTensor。它只表达 inference decoder layer 里最需要的状态：

- hidden state 是否完整。
- hidden state 是否按 token 分片。
- hidden state 是否是 partial sum。
- 这个状态属于哪个 parallel group。

### 6.2 ModuleSpec 把模型子模块标成通信边界

`ModuleSpec` 标注：

- `input_placement`
- `output_placement`
- `kind`: NORM / ATTENTION / DENSE_MLP / MOE / GENERIC
- `call`: attention、MoE、norm 等不同调用约定
- `fusion`: 是否支持 reduce+norm fusion
- `skip_on_idle`

证据：`python/tokenspeed/runtime/models/base/module_spec.py:21-79`

`CompiledDecoderLayer.build_execution_plan()` 会按：

```text
input_layernorm
self_attn
post_attention_layernorm
mlp
```

生成 execution nodes。

attention spec：

```text
input = Replicate(ATTN_TP)
output = Partial(ATTN_TP) if attention TP exists
```

MLP spec：

```text
Dense MLP:
  input = Replicate(DENSE_TP)
  output = Partial(DENSE_TP) if dense TP exists

MoE:
  input = Replicate(MOE_TP_EP)
  output = Partial(MOE_TP_EP) if MoE TP/EP exists
```

证据：`python/tokenspeed/runtime/models/base/decoder_layer.py:237-368`

这就是 local-SPMD 的价值：模型层说“这个模块输入/输出在哪个 group 上是什么 placement”，compiler 再推通信。

### 6.3 Compiler 插入通信的规则

`compile_decoder_layer()` 做的是静态分析：

1. 找到第一个 compute module 的 input group。
2. 根据上一层输出 group 初始化 hidden/residual placement。
3. 对每个 node：
   - norm step 看能否把 partial reduce 融进 norm。
   - compute step 看输入是否需要 all-gather。
   - compute 后看下一层输入 group，决定 all-reduce、reduce-scatter、deferred reduce。
4. 最后返回 `CompiledDecoderLayer(steps, final_placement, mapping)`。

证据：`python/tokenspeed/runtime/models/base/compiler.py:182-260`

中间 compute 的 post-comm：

- 如果下一层 group size 相同，插入 `AllReduceOp`。
- 如果下一层 group size 不同，插入 `ReduceScatterOp`。
- residual 如果需要同步，插入 `ResidualAllGatherOp` 或 `ResidualSliceOp`。
- 如果下一层 norm 能 fuse reduce+norm，可以延迟 reduce。

证据：`python/tokenspeed/runtime/models/base/compiler.py:260-620`

这和 `CommManager` 的思想一致，但 compiler 版本把手写逻辑系统化了。

### 6.4 Compiled runtime

compiler 输出的 `ExecutionStep` 包含：

- `pre_comms`
- `runner`
- `post_comms`
- `captures_aux`
- `skip_on_idle`

runtime forward 时按顺序执行：

```text
for step in steps:
  run pre_comms
  run module
  capture aux if needed
  run post_comms
```

证据：`python/tokenspeed/runtime/models/base/execution.py:70-156`

这让通信不再散落在模型 forward 里，而是成为 execution plan 的一部分。

### 6.5 它和 CommManager 的关系

这一点报告必须说清楚：

**local-SPMD / placement compiler 不是并行策略本身，而是把并行策略安全落到模型图里的机制之一。**

TokenSpeed 目前有两条路径：

| 路径 | 代表 | 特点 |
| --- | --- | --- |
| `CommManager` 手写路径 | Llama、MiniMax M2、Qwen3.5 MoE、DeepSeek V4 custom | 成熟、直接、模型可以深度定制 |
| `CompiledDecoderLayer` compiler 路径 | GPT-OSS | 更系统化，减少手写通信转换 |

证据：

- `python/tokenspeed/runtime/models/base/decoder_layer.py:23-24`
- `python/tokenspeed/runtime/models/gpt_oss.py:440-520`
- `python/tokenspeed/runtime/models/deepseek_v4.py:5515-5668`

DeepSeek V4 并没有走 base compiler，而是自定义 decoder layer，里面仍然直接使用 `CommManager`。

所以 PPT 里不能把“local-SPMD”讲成 TokenSpeed 对 V4 的唯一实现机制。更准确的讲法是：

> 并行策略是“怎么切”；CommManager 和 placement compiler 是“如何把切分落到模型层通信边界”。DeepSeek V4 当前更依赖手写模型路径；compiler 的价值在于未来把这类手写逻辑收敛成可复用的模型层协议。

## 7. 模型层落地：attention、dense、MoE 如何分别吃到 mapping

### 7.1 Attention 使用 `mapping.attn`

DeepSeek V4 attention 构造 layout 时传入 `mapping.attn.tp_size`：

- 检查 `num_attention_heads % attn_tp_size == 0`。
- 计算 `num_local_heads`。
- 根据 compress ratio 选择 SWA / CSA / HCA layout。

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:3049-3120`

V4 attention 的线性层使用：

- `tp_rank=mapping.attn.tp_rank`
- `tp_size=mapping.attn.tp_size`
- `tp_group=mapping.attn.tp_group`

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:4785-4895`

含义：

- attention TP 改变的不只是通信 group，还改变 local head 数、attention cache layout、attention backend 输入形状。
- 对 latent-KV / compressed attention，head 切分和 KV layout 强耦合。
- 因此 vLLM 复制 split attention TP 不是只加参数，还要让 attention backend 和 model loader 都读到 attention group。

### 7.2 Dense 使用 `mapping.dense`

Qwen3.5 MoE 的 dense MLP：

- 如果 `mapping.dense.has_tp`，用 `MergedColumnParallelLinear` 和 `RowParallelLinear`。
- TP group 来自 `mapping.dense.tp_group`。
- 否则用 replicated linear。

证据：`python/tokenspeed/runtime/models/qwen3_5_moe.py:70-137`

DeepSeek V4 的普通 MLP 也根据是否是 shared expert 选择不同 mapping：

- 普通 dense：`mapping.dense`
- shared expert：`mapping.moe.tp_ep_group`

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:3121-3188`

这说明 dense/shared expert 的并行不只是 collective 选择，还影响权重切片和 linear module。

### 7.3 MoE 使用 `mapping.moe`

`MoELayer` 接收：

- `tp_rank`
- `tp_size`
- `ep_rank`
- `ep_size`
- `a2a_backend`

并生成 `MoELayerSpec`，然后由 `select_backend()` 根据硬件架构、量化格式、MoE backend 选择具体实现。

证据：

- `python/tokenspeed/runtime/layers/moe/layer.py:31-193`
- `python/tokenspeed/runtime/layers/moe/core/types.py:37-55`
- `python/tokenspeed/runtime/layers/moe/core/selector.py:27-137`

这意味着 MoE 并行策略落地到三层：

```text
Mapping:
  ranks / groups

MoELayerSpec:
  TP / EP / local experts / all2all backend

Backend:
  flashinfer / triton / cutedsl / mega_moe / DeepEP dispatch-combine
```

vLLM 如果已有 EP 支持，也未必复制了这一层，因为还要看它是否把 backend selection、token dispatch layout、shared expert、quantized expert weight loading 都放在同一个策略里。

### 7.4 Qwen3.5 MoE：常规路径

Qwen3.5 MoE 的 `_forward_tp()` 路径很典型：

1. 本地 token 上做 gate。
2. 用 `CommManager.pre_mlp_comm()` all-gather hidden_states。
3. 同样 all-gather router logits。
4. 做 TopK。
5. 调 MoELayer experts。
6. 输出用 `post_mlp_fused()` reduce-scatter 或 all-reduce 回本地 token。

证据：`python/tokenspeed/runtime/models/qwen3_5_moe.py:151-360`

DeepEP 路径不同：

1. gate 在本地 token 上做。
2. shared expert 本地执行，必要时 dense TP all-reduce。
3. TopK 本地做。
4. experts 内部处理 dispatch / combine。
5. 不走普通 all-gather hidden_states 的 TP path。

证据：`python/tokenspeed/runtime/models/qwen3_5_moe.py:300-360`

这说明 TokenSpeed 的并行策略不仅决定 rank group，还决定模型 forward 的算法路径。

### 7.5 DeepSeek V4：自定义路径

DeepSeek V4 的 decoder layer 自己写了：

- attention 前后的 mHC pre/post。
- V4 attention。
- FFN norm。
- MoE pre-comm 或 mega_moe token count。
- hash MoE input_ids 的额外 all-gather。
- MoE forward。
- post_mlp_comm。

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:5515-5668`

关键点：

1. 如果 `use_mega_moe`，它不走普通 `pre_mlp_comm()`，而是用 `moe_tp_ep_group_scattered_num_tokens(ctx)` 直接计算全局 token 分布。
2. 如果不是 mega_moe，它会先 `pre_mlp_comm()`，再根据 hash MoE 需要对 `input_ids` 做同样的 group all-gather。
3. `post_mlp_comm()` 把 MoE 输出还原到正确 placement。

这不是一个后端插件能单独完成的。V4 的 attention/mHC/MoE/hash router/shared expert/input_ids 通信都在模型层 forward 里交织。

### 7.6 DeepSeek V4 MoE：EP + mega_moe 的硬约束

V4 `DeepseekV4MoE` 使用 `mega_moe` 时要求：

- 必须有 expert parallelism。
- `mapping.moe.tp_size` 必须等于 1。
- 不支持 redundant EP experts。
- `n_routed_experts` 必须能被 `ep_size` 整除。

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:3466-3558`

这再次证明 V4 推荐策略是 EP-only MoE，不是 MoE TP+EP 混合。

V4 `DeepseekV4MegaMoEExperts` 使用 `mapping.moe.tp_ep_group` 获取 symmetric buffer，并调用 DeepGEMM `fp8_fp4_mega_moe`。

证据：`python/tokenspeed/runtime/models/deepseek_v4.py:3227-3715`

迁移到 910C/950DT 时，这部分不是直接可迁移的：

- DeepGEMM / SM100 / CUDA symmetric buffer 不可迁移。
- 但“EP group + fused dispatch/expert compute/combine + shared expert path”的系统接口值得迁移。

## 8. DeepEP / All2All：不是并行策略本身，但决定策略是否跑得快

`DeepEPDispatcher` 有 normal 和 low_latency 两种实现：

- `normal`：更适合 prefill / extend，大 token 量。
- `low_latency`：更适合 decode，小 token 量。
- `auto`：根据 `ForwardMode` 选择 decode 走 low_latency，其他走 normal。

证据：`python/tokenspeed/runtime/layers/moe/dispatcher/deep_ep.py:37-58`, `559-658`

normal dispatch：

- FP8 per-token quant。
- `get_dispatch_layout()` 计算 token/expert/rank 分布。
- `buffer.dispatch()` 得到接收 token、topk、expert token count。
- combine 时用 dispatch handle 合并回原 token。

证据：`python/tokenspeed/runtime/layers/moe/dispatcher/deep_ep.py:230-385`

low_latency dispatch：

- clone contiguous tensor，避免 alias/stride 问题。
- `low_latency_dispatch()` 得到 packed recv hidden、masked_m、event/hook。
- `low_latency_combine()` 合并回原 token。

证据：`python/tokenspeed/runtime/layers/moe/dispatcher/deep_ep.py:398-545`

FP4 CuteDSL executor 更极端：所有 forward 都强制用 DeepEP low_latency，因为 CuteDSL grouped GEMM 需要 low_latency dispatch 产出的 `[num_experts, M_padded, hidden]` 布局。

证据：`python/tokenspeed/runtime/layers/moe/backends/nvfp4/deepep_cutedsl_fp4_executor.py:21-260`

这对我们的判断很关键：

- 并行策略表达决定“应该走 EP”。
- DeepEP/mega_moe 决定“EP 的 token movement 是否足够低延迟”。
- 如果 Ascend 上没有等价 all-to-all / dispatch-combine backend，只移植 Mapping 和 CommManager，性能收益会被 MoE all-to-all 吃掉。

## 9. 这套设计解决的 workload 问题

### 9.1 V4/Kimi 类 MoE 不是单瓶颈模型

这类模型的 decoder step 包含多种不同瓶颈：

| 模块 | 主要瓶颈 | 不适合的统一策略 |
| --- | --- | --- |
| latent-KV / compressed attention | KV layout、local heads、small decode latency、attention backend shape | 强行用全 world TP 可能让 head/KV 切太碎，通信延迟占比上升 |
| dense / shared expert | GEMM compute、activation communication | 完全 DP 会复制计算，吞吐差 |
| routed MoE | expert weight 容量、token dispatch、all-to-all | 单 TP 不能解决专家容量和路由通信 |
| agentic decode | 小 token、rank 间 token 不均、idle rank | padding 式 collective 或固定 batch graph 会拖 p95 |

TokenSpeed 的策略是在模型层允许这些模块使用不同 group。

### 9.2 长上下文 + 多轮 prefix 下，attention 不一定应该跟 dense/MoE 共用 TP

长上下文 decode 的 attention 侧通常受：

- KV cache layout。
- KV read bandwidth。
- local heads。
- compressed/latent-KV reshape。
- small batch kernel launch。

如果 attention TP 被迫等于 dense/MoE TP，可能出现：

- 每 rank local heads 太少，attention kernel occupancy 或 layout 不理想。
- attention output 需要更多跨 rank同步。
- DP replica 数不足，无法吸收多用户 short decode。

V4 文档推荐的 `attention tp=1, dp=4` 就是一个很强的信号：TokenSpeed 不认为 V4 attention 必然应该跟 MoE EP 或 dense TP 共用同一 TP。

### 9.3 短 decode step 下，RSAG 价值大于大 batch prefill

在 prefill 阶段，大 token 量下 all-reduce / all-gather 更接近带宽模型。

在 decode 阶段，每轮只有很少 token：

- collective launch latency 占比高。
- 如果每个 rank token 数不同，padding 浪费明显。
- 如果某些 DP rank idle，仍要保证 collective 进度一致。

TokenSpeed 的 `token_all_gather` / `token_reduce_scatter` 接收 `scattered_num_tokens`，就是为这种 uneven token distribution 服务的。

证据：

- `python/tokenspeed/runtime/distributed/comm_ops.py:304-337`
- `python/tokenspeed/runtime/distributed/comm_manager.py:37-104`
- `python/tokenspeed/runtime/engine/event_loop.py:858-901`

### 9.4 MoE EP 的问题不是“能不能分 expert”，而是 dispatch-combine 能不能低延迟

EP 会减少每 rank 常驻 expert 权重，但引入 token movement。对短 decode 来说，all-to-all latency 很容易成为 p95 主因。

TokenSpeed 的实现把 MoE 策略分成：

- `mapping.moe.ep_size`：专家怎么分。
- `a2a_backend`：token 怎么跨 rank 发。
- `deepep_mode`：prefill/decode 用不同 dispatch 路径。
- `moe_backend`：收到 token 后怎么做 grouped expert GEMM。

这才是“支持 EP”和“EP 有性能”的区别。

## 10. 性能收益模型：不能把几个优化点简单相加

这一章是后续 PPT 必须有的“性能账本”。建议用瓶颈迁移模型，而不是把每项收益直接相加。

### 10.1 单步延迟拆解

可以把一次 decode step 粗略拆成：

```text
T_step =
  T_scheduler_gap
+ T_attention_compute
+ T_attention_comm
+ T_dense_compute
+ T_dense_comm
+ T_moe_dispatch
+ T_moe_compute
+ T_moe_combine
+ T_sampling/output
```

更准确地说，部分 compute/comm 可 overlap，所以应该看 critical path：

```text
T_step ~= max(
  attention path,
  dense path,
  moe dispatch + expert compute + combine path
) + non-overlapped CPU/runtime gap
```

Scheduler/KV 深挖回答的是 `T_scheduler_gap`、KV reuse、overlap。

并行策略深挖回答的是：

- `T_attention_comm` 是否被不必要 TP 放大。
- `T_dense_compute/T_dense_comm` 是否用合适 TP。
- `T_moe_dispatch/T_moe_combine` 是否用 EP/DeepEP/mega_moe 降低。
- DP replica 是否提高并发吞吐而不拉高 p95。

### 10.2 并行策略带来的收益路径

| 收益路径 | 机制 | 主要影响 | 风险 |
| --- | --- | --- | --- |
| attention 与 dense/MoE 拆组 | attention 用小 TP/DP，dense/MoE 用 TP/EP | 降低 attention small-step 通信，提升多用户 decode 利用率 | dense/MLP 前后需要更多 gather/scatter |
| AR -> RSAG | group size 不同时用 token reduce-scatter/all-gather | 避免把所有 rank 都保留完整 token，减少后续局部 token 工作量 | small-message RSAG launch 可能比 AR 慢 |
| DP token-aware metadata | forward 前 gather 全局 token 数 | uneven decode 下减少 padding/shape 错误，支持 idle forward | CPU all-gather 有固定开销 |
| MoE EP + DeepEP | expert weight 分布 + low-latency dispatch | 降低 expert 常驻权重压力，改善 decode p95 | Ascend 上 DeepEP 不可直接迁移 |
| fused reduce+norm | all-reduce 与 RMSNorm 融合 | 小 token decode 降低 launch 和读写 | split group 后可能被禁用 |

### 10.3 估算区间

如果 vLLM-Ascend baseline 已经有等价并行策略和等价 all-to-all backend，TokenSpeed 这部分增量可能只有：

```text
0-10%
```

主要来自：

- 更细 token-aware collective。
- 更好的 residual placement/fusion。
- 某些模型 patch 的更优组织。

如果 baseline 是 uniform TP/DP，缺少 layer-family split，TokenSpeed 的增量可能是：

```text
10-30%
```

来源：

- attention 不再被 dense/MoE 的 TP 形态绑架。
- MoE EP 路径更贴近 routed expert 的真实瓶颈。
- decode 下减少 padding collective 和 idle rank 影响。

如果 baseline 没有等价低延迟 MoE dispatch-combine，理论差距可以超过 30%，但这时收益不应全部归功于“并行策略实现”，而是 MoE backend / all-to-all / kernel 共同造成。

因此 PPT 里建议用这样的归因：

```text
Scheduler/KV:
  10-25%，看 prefix reuse、KV 压力、p95 gap。

Parallel strategy implementation:
  10-30%，看 baseline 是否已有 layer-family split + token-aware comm + EP backend。

Placement compiler / local-SPMD:
  5-15% 直接性能收益，更大的价值是降低复杂策略落地成本。

MLA / latent-KV kernel execution:
  低权重介绍；vLLM 已支持 TokenSpeed 算子后，单 kernel 护城河下降。
```

### 10.4 需要验证的性能信号

这里不是研究矩阵里的主列，而是性能归因时必须采的信号：

- 每 step collective 次数：all-reduce / all-gather / reduce-scatter / all-to-all。
- 每类 collective 的 bytes 和 latency，按 attention/dense/MoE group 拆开。
- rank token 分布：`global_num_tokens` 的 max/min/zero-rank 数。
- MoE dispatch 后每 rank recv tokens、每 expert tokens、masked_m。
- prefill 与 decode 分开统计。
- p50/p95 ITL、TPM、TPS/User。
- attention local head 数与 kernel occupancy。
- dense/MoE GEMM shape 分布。
- fusion 命中率：allreduce+norm 是否因 split group 被禁用。

## 11. vLLM/vLLM-Ascend 复制难度判断

### 11.1 Config-level 可复制

容易复制的是参数面：

- `--attn-tp-size`
- `--dense-tp-size`
- `--moe-tp-size`
- `--data-parallel-size`
- `--enable-expert-parallel`
- `--moe-backend`
- `--all2all-backend`

这只是入口，不构成护城河。

### 11.2 Model-patch-level 可复制

中等难度的是让模型模块读不同 mapping：

- attention linear/head/KV layout 读 `mapping.attn`。
- dense linear 读 `mapping.dense`。
- MoE layer 读 `mapping.moe`。
- checkpoint loader 按 EP rank 加载 local experts。
- LM head / logits processor 按 attention TP 或 DP 处理。

这需要逐模型适配，但不是不可复制。

TokenSpeed 的 DeepSeek V4 已经把这些 patch 写得很深：

- attention layout 与 `attn_tp_size` 强绑定。
- MoE `mega_moe` 强绑定 EP-only。
- hash MoE 需要 `input_ids` 跟 hidden 一起按 MoE group gather。
- shared experts 要走额外 all-gather/reduce-scatter。

这部分 vLLM 可以复制，但代价是 model implementation 级别。

### 11.3 Architecture-level 复制

最难复制的是模型执行协议：

1. Runtime 有多套 process group，不是一个 global TP。
2. `ForwardContext` 携带 DP token distribution。
3. attention/dense/MoE 边界有统一 placement 语义。
4. residual placement 被系统追踪。
5. all-reduce 与 RSAG 的切换规则和 fusion 规则统一。
6. MoE backend selection 和 all-to-all dispatch layout 是并行策略的一部分。
7. CUDA graph / eager path / idle forward / decode padding 都必须和 token-aware collective 一起工作。

如果 vLLM/vLLM-Ascend 当前架构强假设单一 `tensor_parallel_size` 贯穿模型，则复制 TokenSpeed 这部分是 architecture-level。

如果 vLLM/vLLM-Ascend 已经有类似多 group execution graph，则复制降为 model patch + backend patch。

但无论如何，它不是“把 TokenSpeed MLA 算子接进来”就能复制的。

## 12. 对 910C + 950DT 的迁移含义

### 12.1 可迁移的部分

适合迁移的是抽象和执行协议：

- attention/dense/MoE 分离 mapping。
- process group 按 layer family 组织。
- hidden/residual placement 的通信插入规则。
- DP token metadata gather。
- token-aware all-gather/reduce-scatter 接口。
- MoE EP dispatch-combine 的 runtime interface。
- 模型层把 attention layout、dense linear、MoE experts 分别接到不同 mapping。

这些不依赖 CUDA，本质是系统结构。

### 12.2 需要重做的部分

不能直接迁移的是后端实现：

- NCCL process group -> HCCL process group。
- TritonRSAG / CUDA token collective -> Ascend 自定义 collective 或 HCCL wrapper。
- DeepEP -> Ascend all-to-all / dispatch-combine。
- DeepGEMM mega_moe -> Ascend grouped expert GEMM。
- fused allreduce+RMSNorm -> Ascend 融合算子。
- CUDA stream / event semantics -> Ascend runtime stream/event。

这意味着在 910C/950DT 上，TokenSpeed 的“并行策略实现”不是开箱即用收益，而是提供了一套需要重建 backend 的 execution plan。

### 12.3 910C 与 950DT 的不同验证目标

910C 更适合验证：

- Mapping/CommManager/placement compiler 能否在 Ascend runtime 表达出来。
- HCCL group 与 uneven token collective 是否可行。
- 模型 correctness。
- 小规模 p95 是否不退化。

950DT 更适合验证：

- EP all-to-all 是否有足够带宽/低延迟。
- MoE grouped GEMM 是否能接近硬件上限。
- attention DP + dense TP + MoE EP 组合是否真的优于 uniform TP。
- agentic workload 的 TPM/GPU 和 p95 是否拉开。

## 13. PoC 建议：并行策略这章怎么实证

### 13.1 最小实验矩阵

先不要平均测试所有组合。建议围绕 V4/Kimi workload 做 4 组：

| 实验 | attention | dense/shared | MoE | 目的 |
| --- | --- | --- | --- | --- |
| A baseline | uniform TP | uniform TP | TP 或 EP | 模拟传统单策略 |
| B split no EP | attention 小 TP/DP | dense TP | MoE TP | 看 layer-family split 本身 |
| C split + EP | attention DP/小 TP | dense TP | MoE EP | 看 V4/Kimi 核心策略 |
| D split + EP + low-latency all2all | 同 C | 同 C | EP + optimized dispatch | 看真正性能上限 |

### 13.2 必须做 ablation

为了避免把收益误归因给单个 kernel：

- 固定 MoE backend，只切 uniform vs split mapping。
- 固定 mapping，只切 normal vs low_latency all-to-all。
- 固定 mapping/backend，只切 allreduce fusion。
- 固定请求分布，分别测 prefill-heavy 和 decode-heavy。
- 单独测 rank token imbalance 对 p95 的影响。

### 13.3 胜负线

并行策略这一章可以设单独胜负线：

```text
相比 vLLM-Ascend baseline:

1. decode-heavy agentic workload:
   p95 ITL 不退化，TPM/GPU +10-20% 起步。

2. MoE-heavy workload:
   MoE dispatch+combine 占 step latency 的比例下降。

3. 长上下文多轮 prefix:
   attention local layout 不因 split strategy 导致质量/正确性问题。

4. 综合目标:
   与 Scheduler/KV 叠加后，整系统 TPM/GPU +20-30%，TPS/User 不退化。
```

如果只看到单个 MLA/kernel 带来的收益，而并行策略 ablation 没有收益，则应该优先反哺 vLLM-Ascend，而不是完整适配 TokenSpeed。

## 14. 本章对 PPT 大纲的改写建议

原先“并行策略实现”和“local-SPMD”要拆开，但关系要讲准：

### 第二章：并行策略实现

重点讲：

- `Mapping(attn/dense/moe)` 如何从参数变成 runtime topology。
- V4 文档推荐策略如何推导成 attention DP=4、dense TP=4、MoE EP=4。
- `CommManager` 如何在 layer family 边界做 AR / RSAG / residual placement。
- DeepSeek V4 如何手写接入 attention layout、hash MoE、mega_moe。
- 性能收益来自瓶颈匹配和减少不必要 collective，而不是“支持 EP”。

### 第三章：local-SPMD / placement compiler

重点讲：

- 它是把策略安全落图的机制，不是策略本身。
- 当前不是所有模型都用 compiler，DeepSeek V4 仍是手写路径。
- 价值在降低后续复杂模型适配成本，把 CommManager 的经验固化成静态分析。
- 直接性能收益低于 Scheduler/KV 和并行策略，但对可复制性和演进速度重要。

## 15. 结论

本章结论：

1. TokenSpeed 的并行策略实现是核心护城河候选，但护城河不在“flag 多”，而在 `Mapping -> process group -> model weights/layout -> communication placement -> backend` 的完整链路。
2. DeepSeek V4 的推荐配置是最强证据：attention DP、dense/full TP、MoE EP、mega_moe backend 共同组成 execution plan。
3. TokenSpeed 当前并不支持任意 MoE TP+EP 组合；报告中必须诚实写出这个边界。
4. local-SPMD / placement compiler 的价值是把 CommManager 手写通信模式系统化，但当前 DeepSeek V4 不走 compiler，所以不能把它包装成 V4 性能收益的主要来源。
5. vLLM/vLLM-Ascend 复制参数面很容易，复制模型 patch 中等难度，复制 token-aware multi-group execution protocol 是 architecture-level。
6. 在 910C/950DT 上，真正需要验证的是：Ascend 是否能重建 token-aware collective、MoE dispatch-combine、grouped expert GEMM 和 fused norm，使这套 execution plan 真的转化为 p95/TPM 收益。
