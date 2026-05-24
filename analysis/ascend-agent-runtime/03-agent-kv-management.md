# TokenSpeed Agent 场景 KV 管理优化深挖 v0.1

本文回答一个更具体的问题：

> TokenSpeed 为 agentic workload 下的 KV 管理做了哪些优化？这些优化具体怎么在代码里实现，为什么对多轮 agent 场景有意义，vLLM / vLLM-Ascend 是否容易迁移复制？

这里不再把 KV 管理写成“有 prefix cache / 有 C++ scheduler”的功能清单。更准确的理解是：

```text
TokenSpeed 把 KV page 的生命周期变成 request FSM 的一部分。

请求不是简单 waiting/running/finished；
KV page 也不是简单 alloc/free。

每个请求状态都携带它当前持有的资源：
prefix tree node ref、request-local allocator、req pool slot、
host/device page pair、Mamba checkpoint、paged cache group table。
```

这套设计对 agent 场景的价值，不在平均吞吐的常规 benchmark 里最明显，而是在 repeated prefix、短 decode step、请求 churn、高 KV pressure、DP rank cache footprint 不均时，降低重复 prefill、降低无效 decode、减少队列阻塞和 p95/p99 tail。

## 1. Agent 场景下 KV 管理的核心矛盾

Agent workload 通常不是单个长 prompt 一次性跑完，而是一个 session 在多个轮次中反复经历：

```text
用户输入 / system prompt / tool schema / history prefix
  -> 模型生成短文本
  -> tool call
  -> tool result append 到上下文
  -> 再次发起短 decode
  -> 可能取消、超时、grammar 约束、并行采样
```

因此 KV 管理面对的是几个相互冲突的目标：

| 压力 | KV 层需要解决的问题 | 如果处理不好 |
|---|---|---|
| 多轮 repeated prefix | 历史 prefix 如何安全复用，finish 后生成的 KV 何时进入 cache | 重复 prefill，TTFT 上升 |
| 短 decode step | 每轮只生成 1 个或少量 token，tail page 和 reserve 很细碎 | page 浪费、频繁 admission 失败、p95 ITL 抖动 |
| 请求 churn | abort / finish / grammar-failed admission 很频繁 | 已取消请求继续占 req slot / KV page |
| KV pressure | 长 session 常驻 KV，短请求还要插进来 | 全局队列阻塞、preemption 后重算 |
| DP cache skew | 某个 DP rank 积累很多长 session | rank 间 page usage 不均，引发局部 retraction |
| async cache movement | host/device copy 与 forward 重叠 | transfer 期间 early-free / double-use / stale page |

所以 TokenSpeed 的 KV 优化不是单点，而是一条闭环：

```text
prefix match
  -> page allocation / req pool admission
  -> prefill chunk window
  -> full pages move into prefix tree
  -> decode reserve
  -> output feedback updates reserve / finish
  -> finish writeback or retract writeback
  -> later loadback / recovery
```

## 2. 对照一：KV ownership 写进 C++ Request FSM，而不是停留在 block manager 语义

### 2.1 TokenSpeed 具体实现

TokenSpeed 的 C++ request state 不是普通枚举。状态对象直接携带资源所有权：

- `Submitted` 只有 token container 和 page size，不持有 allocator。源码：`tokenspeed-scheduler/csrc/fsm/forward_states.h:58-76`。
- `BaseState` 持有 `DeviceNodeRef`、`LocalKVAllocator`、可选 `LocalMambaAllocator`。源码：`forward_states.h:96-145`。
- `ForwardState` 增加 `ReqPoolIndex`，并通过 `GetOccupiedPages()` 返回 prefix tree pages + local tail pages。源码：`forward_states.h:148-168`。
- `Prefilling / PrefillDone / Decoding` 都持有 host/device node ref、local KV allocator、req pool index、reserve token 信息。源码：`forward_states.h:171-277`。
- `Draining / WritingBack` 持有具体 writeback page pairs，并用 node refs pin 住 host/device pages。源码：`forward_states.h:279-322`。
- `Retracting / Retracted` 保留 token container、host ref、local allocator，用于后续 recovery。源码：`forward_states.h:324-395`。

底层 page ownership 由 `OwnedPages` 管理：

- move-only，禁止 copy；
- 析构时如果仍持有 allocator 和 page ids，会自动 Deallocate；
- `TakeFirst / TakeLast / Append / Detach / ReleaseOwnershipByID` 用于拆分、合并、移交 page ownership。

源码：`tokenspeed-scheduler/csrc/resource/allocator/owned_pages.h:31-68`，`owned_pages.cpp:30-86`。

这意味着 KV page 的生命周期不是“谁记得释放谁释放”，而是被 C++ 类型系统和 move 语义管住。

### 2.2 vLLM / vLLM-Ascend 对照基线

这里的对照对象不是一个抽象的“普通 cache manager”，而是 vLLM V1 当前主线的 scheduler / KV cache 结构。本文对照的是 vLLM 主线快照 `9640970` 中的 V1 scheduler/KV 代码，只作为必要 baseline，不展开成 vLLM 深挖。

- `vllm/v1/core/sched/scheduler.py` 的 `Scheduler.schedule()` 负责请求队列、token budget、chunked prefill、spec decode、preemption，并输出 `SchedulerOutput`。
- `vllm/v1/core/kv_cache_manager.py` 的 `KVCacheManager` 负责 `get_computed_blocks()`、`allocate_slots()`、`cache_blocks()`、`free()` 和 prefix cache stats。
- `vllm/v1/core/block_pool.py` 的 `BlockPool` 管理 `KVCacheBlock`、free block queue、prefix-cache hash map。
- `vllm/v1/core/kv_cache_utils.py` 的 `KVCacheBlock` 主要持有 `block_id`、`ref_cnt`、`block_hash` 和 free-list 指针。
- 当 `allocate_slots()` 返回 `None` 时，vLLM scheduler 会 preempt running request；`_preempt_request()` 调用 `kv_cache_manager.free(request)`，把 request 状态设为 `PREEMPTED`，并把 `num_computed_tokens` 置 0 后放回 waiting queue。

这个 baseline 说明 vLLM 并不是“没有 prefix cache / 没有 preemption”。它有成熟的 block manager、prefix hash、KV block ref count、preemption、external KV connector 入口和 MLA/DeepSeek V4 KV cache spec。本文要比较的不是功能有无，而是语义边界：

| 维度 | vLLM V1 baseline | TokenSpeed 当前实现 |
|---|---|---|
| 请求状态 | scheduler 维护 waiting/running/preempted/finished 等请求状态 | C++ Request FSM 状态对象直接携带 KV page / node ref / req pool / transfer pairs |
| KV ownership | `KVCacheManager` / `BlockPool` 管 block 分配、prefix hash、ref count | `OwnedPages`、`LocalKVAllocator`、`DeviceNodeRef` / `HostNodeRef` 随 FSM transition 迁移 |
| prefix cache | `get_computed_blocks()` 找最长 cached blocks，`cache_blocks()` 缓存 full blocks | `schedulePrefillFirstChunk()` 同时区分 device depth / host depth / loadback diff |
| preemption | `_preempt_request()` free KV blocks，request 回 waiting，后续按 prefix/cache 重算或命中恢复 | `ScheduleRetractEvent` 将可恢复 KV 转成 host/device tree state，进入 `Retracting/Retracted` |
| cache movement | 通过 KV connector / P-D / external computed tokens 等机制表达外部 KV | Scheduler 同一轮 `ExecutionPlan` 同时返回 `ForwardOp + CacheOp`，cache event 再推进 FSM |

所以，TokenSpeed 的潜在优势不是“vLLM 没有 KV cache”，而是它把更多 KV 生命周期语义放进 scheduler FSM 和 execution plan 里。如果 vLLM/vLLM-Ascend 要复制这一层，需要改的是 scheduler、KV cache manager、worker loop 和 cache transfer 事件之间的协议，而不只是增加一个 prefix cache 开关。

### 2.3 为什么对 agent 场景重要

Agent 场景里，一个请求很容易处于“半完成、可继续、可取消、可恢复”的状态。例如 tool call 之后会再次携带历史 prefix；长 session 可能在 KV 紧张时被暂时挪走；用户取消时不能再继续烧 decode step。

如果 KV page ownership 没有和 request lifecycle 绑定，最容易出的问题是：

- 已经进入 prefix tree 的 page 被 request-local allocator 析构释放；
- async writeback 还在读 device page，但 scheduler 已经把 page 分配给新请求；
- retract 后请求恢复时，req pool slot 和 valid cache length 对不上；
- abort / finish 只改前端状态，C++ scheduler 中的 KV 资源没有及时释放。

TokenSpeed 的做法是把这些边界都变成状态转移：

```text
Submitted
  -> Prefilling / PrefillDone
  -> Decoding
  -> Draining -> WritingBack -> Finished
  -> Retracting -> Retracted -> Decoding
```

这就是它相对 vLLM V1 block-manager baseline 的核心差异：**TokenSpeed 的 FSM 状态本身就是资源所有者**。vLLM 的 block manager 已经很强，但 request lifecycle 与“device/host node ref、request-local allocator、writeback pairs、retract recovery”之间不是同一种状态机表达。

### 2.4 vLLM / vLLM-Ascend 复制难度

单独复制一个 prefix cache 或 page allocator 不难；难的是把 request state、KV block manager、async transfer、preemption/recovery、output feedback 都改成同一套 ownership protocol。

复制等级：

| 复制对象 | 难度 | 判断 |
|---|---:|---|
| prefix cache / block hash | 中 | vLLM 已有同类能力 |
| request-local allocator + prefix-tree ownership transfer | 中高 | 需要改 KV block 生命周期 |
| Draining/WritingBack/Retracting/Retracted 状态 | 高 | 触碰 scheduler、KV manager、worker loop |
| 完整 request FSM + RAII ownership | 高 | 架构级复制，不是 config patch |

## 3. 对照二：prefix reuse 同时看 device / host depth，并把 loadback 编入调度计划

### 3.1 具体实现

首次 prefill 调度时，`Scheduler::schedulePrefillFirstChunk()` 会对请求的 full paged tokens 做 prefix match：

```cpp
MatchResult match_result =
  hybrid_prefix_cache_
    ? hybrid_prefix_cache_->Match(request->GetFullPagedTokens(true))
    : kv_prefix_cache_.Match(request->GetFullPagedTokens(true));

device_matched = match_result.device.DepthInPage();
host_matched   = match_result.host.DepthInPage();
```

源码：`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:67-87`。

这里的关键不是“命中多少 prefix”，而是它同时区分：

- device cache 已命中的 page depth；
- host cache 已命中的 page depth；
- host 有、device 没有的 `loadback_diff`。

当 host 命中比 device 更深时，scheduler 会：

1. 计算需要 loadback 的 tokens；
2. 先确保 device capacity；
3. 为 host-only nodes 分配 device resource；
4. 生成 `LoadBackOperation`，把 host page 搬回 device page。

相关代码：

- `forward.cpp:82-87`：计算 `loadback_diff`、`loadback_tokens`、`unscheduled`。
- `forward.cpp:97-104`：把 loadback tokens + 本轮 prefill tokens + decode reserve 合并进 device page admission。
- `forward.cpp:530-538`：如果 `loadback_diff` 非空，生成 LoadBackOp。
- `forward.cpp:310-320`：从 diff nodes 构造具体 `(host_page, device_page)` pairs。
- `forward_events.cpp:84-145`：`Submitted -> Prefilling/PrefillDone` 转移中，如果 host depth 更深，会 `AllocateResourceOfType<Device>` 并把 `DeviceNodeRef` 指向 host last node。

这说明 TokenSpeed 不是等 forward 执行时才发现缺 KV，而是在 scheduler 生成 plan 时就把 cache movement 编进去。

### 3.2 相对 vLLM baseline 的差异

多轮 agent session 的典型情况是：prefix 很长，新增 token 很少。真正影响 TTFT 的不是 prompt 总长度，而是“重复 prefix 能不能少算、能不能及时回到 device”。

相对 vLLM V1 的 `KVCacheManager.get_computed_blocks()` 返回本地 prefix-cache computed blocks、`allocate_slots()` 为新 tokens 分配 slots、外部 KV 通过 connector / external computed tokens 进入调度的 baseline，TokenSpeed 这一段的潜在优势在于：它在同一个 C++ schedule decision 中同时处理 device prefix、host prefix、loadback、page admission 和本轮 forward window。

```text
同一次 schedule 决策里同时回答：
  device 已经有多少 prefix？
  host 还保存多少 prefix？
  需要 loadback 多少 page？
  loadback + 新 prefill + decode reserve 是否放得下？
  本轮应该 forward 哪段 window？
```

因此 repeated prefix 的收益不会停留在“cache tree 可以 match”，而是会落到 execution plan：

```text
ForwardOp: 只计算未命中的 prefill window
CacheOp:   同时提交 host -> device loadback
```

这个差异不是说 vLLM 不能做外部 KV 或 P/D KV transfer，而是说 TokenSpeed 把 device/host cache depth 和 loadback diff 作为 scheduler FSM 的一等输入，并把 loadback 与 forward 放进同一个 `ExecutionPlan`。如果 vLLM/vLLM-Ascend 要复制，需要把 connector / cache transfer 的完成事件与 scheduler block lifecycle 更紧地合并，而不是只在 backend 层补一个拷贝函数。

### 3.3 需要注意的限制

当前 DeepSeek V4 baseline 对 hierarchical kvstore 明确有限制：

- `event_loop.py:237-246` 判断 `DeepseekV4TokenToKVPool` 后，如果 `enable_kvstore` 为真，会抛 `NotImplementedError`，要求 `--disable-kvstore`。
- 因此，**host/L2/L3 loadback/writeback 不能直接作为 DeepSeek V4 当前可用收益**。

更准确的正式表述应是：

```text
TokenSpeed 的通用 runtime 已经把 device/host/L3 KV movement 编入 scheduler plan；
但 DeepSeek V4 当前路径禁用 hierarchical kvstore。
对架构竞争力判断，短期应优先看 device prefix reuse、page ownership、
decode reserve、retract 语义和 DP cache-aware dispatch 是否构成完整 runtime 协议；
host/L2/L3 是后续适配目标，不应当被提前记入收益。
```

## 4. 优化三：request-local tail page 与 prefix-tree full page 分离

### 4.1 具体实现

TokenSpeed 把一个请求当前占用的 KV page 拆成两类：

1. prefix tree 已拥有的完整 pages；
2. request-local allocator 持有的 tail pages。

`PageContainer` 把二者合并成 forward 可见的 page list：

```cpp
std::vector<std::int32_t> prefix_pages = DevicePagesFromRoot(node_);
std::vector<std::int32_t> local_pages = local_kv_allocator_->Pages();
prefix_pages.insert(prefix_pages.end(), local_pages.begin(), local_pages.end());
```

源码：`tokenspeed-scheduler/csrc/resource/page_container.cpp:30-35`。

`LocalKVAllocator` 负责 request-local pages 和 tail capacity：

- `Acquire(num_tokens)` 根据 tail page 剩余空间决定是否申请新 page；
- `TailPageAvailableTokens()` 返回 tail page 剩余 token 数；
- `TakeFullPages()` 可以把已满 pages 从 request-local allocator 移走，只保留 tail page。

源码：`tokenspeed-scheduler/csrc/resource/allocator/kv_allocator.h:32-58`，`kv_allocator.cpp:31-67`。

prefix tree 接管 page 的关键点在 `KVPrefixCache::Insert()`：

- 将 prefix pages 和 allocator pages 合成 page ids；
- walk radix tree 找 mismatch；
- 对没有资源的 node，从 `OwnedPages` 中 `TakeLast(node_num_pages)`；
- `node->AttachResource(std::make_unique<NodeResource<RType>>(std::move(node_owned)))`。

源码：`tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp:199-315`。

### 4.2 为什么这对短 decode 很关键

Agent 的 decode step 往往很短，很多请求每轮只新增 1 个 token。此时如果每个 step 都把部分填充的 page 当成全局 prefix cache page 处理，会带来两个问题：

- prefix tree 中混入大量尚未完整、未来还要继续写的 tail pages；
- page ownership 在 request-local append 和 prefix-cache reuse 之间反复摇摆。

TokenSpeed 的策略是：

```text
完整 page 可以进入 prefix tree；
尾页继续由 request-local allocator 持有；
forward 时 PageContainer 把 tree prefix + local tail 拼成当前 block table。
```

这让短 decode 的 KV append 更稳定：tail page 不急着进入全局 cache，但仍能参与当前请求的 attention 读取。

### 4.3 性能收益路径

这类优化不一定显著提升平均 tokens/s，但会影响：

- tail page waste；
- device page admission failure；
- active KV pages / total KV pages；
- decode p95 ITL；
- retract 触发次数。

对高并发短 decode，如果 baseline 因 tail page 管理粗糙导致 page pressure，收益可能体现在 p95/p99，而不是 p50。

## 5. 优化四：decode reserve 由 output feedback 动态更新，避免固定过度预留

### 5.1 具体实现

decode 调度前，TokenSpeed 会检查当前 tail page 是否足够容纳下一轮 decode reserve：

```cpp
tail_available = request->TailPageAvailableTokens();
extra_tokens = max(0, reserve_num_tokens_in_next_schedule_event - tail_available);
pages_needed = ceil(extra_tokens / page_size);
EnsureCapacityByEvict(Device);
checkPagedCacheGroupAdmission(...);
```

源码：`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:162-181`。

reserve 不是固定值，而是由 Python output processor 在每轮结果出来后回灌：

- `generation_output_processor.py:602-605`：decode slot 未完成时，发送 `UpdateReserveNumTokens`。
- `scheduler_utils.py:145-149`：构造 `ForwardEvent.UpdateReserveNumTokens`。
- `forward_states.h:263-268`：`Decoding` 状态保存并更新 `reserve_num_tokens_in_next_schedule_event`。
- `forward_events.cpp:224-238`：下一次 `ScheduleDecodeEvent` 按 reserve acquire pages。

### 5.2 Agent 场景为什么需要动态 reserve

普通 decode 每轮 1 token 时，reserve 近似固定。但 agent runtime 常常叠加：

- speculative decode；
- grammar / structured output；
- early stop；
- tool call 触发 finish；
- 不同请求输出长度分布差异大。

如果静态按最大 draft width 或最大输出长度预留，KV page 会被长尾请求锁住；如果预留太少，则 forward 过程中可能才发现 page 不够。TokenSpeed 的做法是把“上一轮实际接受了多少 token”作为下一轮 reserve 的输入，形成闭环：

```text
GPU output length
  -> OutputProcessor
  -> UpdateReserveNumTokens
  -> Scheduler Decoding state
  -> next scheduleDecode admission
```

### 5.3 性能收益路径

主要 counter：

- reserve tokens vs actual accepted tokens；
- tail page available tokens；
- admission failure count；
- page allocation per decode step；
- spec decode accept length；
- p95 ITL under short decode。

这个优化本身可能不是 20% 级别吞吐提升，但能降低短 decode 场景的 page pressure 和 p95 抖动。

## 6. 优化五：finish 不是释放，而是把生成结果转成下一轮可复用 prefix

### 6.1 具体实现

请求完成时，TokenSpeed 不只是释放 pages。`FinishEvent::apply()` 会：

1. 取出当前 request 的 full paged tokens；
2. 找到 prefix tree 已有的 device pages；
3. 将 request-local pages 插入 device prefix cache；
4. 如果 L2/host cache 开启且 host depth 落后 device depth，则分配 host resource；
5. 进入 `Draining`，后续生成 writeback op；
6. writeback 完成后进入 `Finished`。

源码：

- `forward_events.cpp:282-333`：Finish 时插入 device prefix cache，并决定是否进入 Draining。
- `forward_events.cpp:351-358`：`Draining -> WritingBack`。
- `forward_events.cpp:361-365`：`WritingBack -> Finished`。
- `scheduler.cpp:393-415`：Draining 请求生成 `WriteBackOperation`。
- `scheduler.cpp:424-429`：Finished 请求释放 paged cache group 并从 request map 删除。

`Draining` 和 `WritingBack` 的关键不是状态名，而是它们持有 concrete page pairs 和 RAII node refs：

- `Draining` 注释说明 page pairs 在状态转移时捕获，避免后续 radix tree split 改变 node/page 映射。源码：`forward_states.h:279-305`。
- `WritingBack` 持有 device/host node refs，让 async copy 期间 pages 不可 evict。源码：`forward_states.h:307-322`。

### 6.2 Agent 场景为什么重要

Agent 多轮对话中，“本轮生成结果”会变成“下一轮 prompt prefix”的一部分。finish 时如果只是释放 KV，那么下一轮就必须重新 prefill 历史上下文。

TokenSpeed 的 finish 路径等价于把完成请求的完整 pages 变成可复用的 prefix-tree-owned cache：

```text
request-local generated KV
  -> FinishEvent
  -> KVPrefixCache::Insert(Device)
  -> optional Device -> Host writeback
  -> Finished
```

这对 multi-turn agent 的收益路径很清晰：

- 下一轮 prompt 命中 device prefix：减少 prefill compute，降低 TTFT。
- 下一轮 prompt 只命中 host prefix：通过 loadback 规避全量 recompute。
- cache pressure 时，tree-owned pages 可以按 LRU / refcount 参与 eviction，而不是挂在已完成 request 上。

## 7. 优化六：KV pressure 下不是简单 preempt，而是 retract / loadback 的可恢复退让

这是 TokenSpeed KV 管理里最值得深挖的 agent 优化点。

### 7.1 触发条件

在 `Scheduler::newForwardOperation()` 中，如果本轮没有任何 forward op 能调度，并且存在 active decode / prefill-done 请求，TokenSpeed 会选择 token size 最大的 active request 做 retract：

```cpp
if (ops.empty() && !candidates.empty()) {
  retract_candidates = Decoding or PrefillDone;
  victim = max_element(TokenSize);
  newRetractOperation(victim);
}
```

源码：`tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:561-579`。

这个策略很朴素，但它解决的是 agent 场景下的真实问题：长 session 占用大量 device KV，导致短请求或其他 session 无法进入。与其让所有请求阻塞，不如把最长的 active request 转成可恢复状态。

### 7.2 Retract 具体做了什么

`Scheduler::scheduleRetract()` 的流程：

1. 取请求 full paged tokens；
2. 获取当前 prefix tree 中已有的 prefix pages；
3. 从 request-local allocator 拿出可插入的 pages；
4. 将这些 pages 插入 device prefix cache；
5. 重新用 `MatchIntent::StateRecovery` 做 match；
6. 如果 device depth > host depth，则为 host 分配缺失 pages；
7. 返回 `ScheduleRetractEvent`。

源码：`forward.cpp:275-307`。

状态转移 `ScheduleRetractEvent::applyRetract()` 会：

- 为 device-only nodes 分配 host pages；
- 构造 concrete `(device_page, host_page)` pairs；
- 移走 local KV allocator；
- 如果有 Mamba checkpoint / working state，把它插入 hybrid prefix cache；
- 返回 `Retracting` 状态。

源码：`forward_events.cpp:419-475`。

随后 `applyEventAndGenerateOp(ScheduleRetractEvent)` 读取状态中预先捕获的 page pairs，生成 `WriteBackOperation`。源码：`forward.cpp:323-345`。

writeback 完成后，`WriteBackDoneEvent(Retracting)` 进入 `Retracted`：

- 释放 DeviceNodeRef；
- 保留 HostNodeRef；
- 保留 LocalKVAllocator；
- 保留 token container；
- 请求可以后续恢复。

源码：`forward_events.cpp:368-376`。

### 7.3 Recovery 具体做了什么

当资源允许，`scheduleDecodeFromRetracted()` 会把 `Retracted` 请求恢复：

1. 使用 `MatchIntent::StateRecovery` 重新 match full paged tokens；
2. 计算 host depth 和 device depth；
3. 对 host-only nodes 分配 device pages；
4. 生成 `LoadBackOperation`；
5. 重新分配 req pool slot；
6. local allocator acquire decode input tokens；
7. 进入 `Decoding`。

源码：

- `forward.cpp:184-272`：recovery admission、loadback diff、device capacity、paged cache group admission。
- `forward_events.cpp:240-279`：`Retracted -> Decoding`。
- `forward.cpp:456-489`：生成恢复后的 decode op，设置 `hist_token_len`，重建 paged cache pages。

Python 侧 `ModelExecutor.reset_valid_cache_length()` 会处理 retraction recovery：

- 如果 `forward_op.hist_token_lens` 不是全 `-1`，说明 scheduler 要覆盖 valid cache length；
- 它把 `runtime_states.valid_cache_lengths[pool_idx]` 设置为历史 token length；
- 这样 `out_cache_loc` 和 position ids 能从恢复后的 KV 长度继续。

源码：`python/tokenspeed/runtime/execution/model_executor.py:732-801`。

### 7.4 这和普通 preemption 的差异

普通 preemption 往往是：

```text
KV 不够
  -> 把 request 移出 running
  -> 释放 block
  -> 后续依赖 prefix cache 或 recompute 再回来
```

TokenSpeed retract 更像：

```text
KV 不够
  -> 把 active request 的已完成 KV 转成 prefix-tree-owned resource
  -> 需要时写回 host
  -> request 进入 Retracted，可恢复
  -> later host/device match + loadback + hist_token_len 修正
  -> 继续 decode
```

差异不在“是否能把请求停下来”，而在**停下来以后，它的 KV 语义是否仍然完整可恢复**。

### 7.5 性能收益路径

Retract/recovery 的收益只在 KV pressure 真实存在时出现：

- 降低全局队列阻塞；
- 降低长请求占满 device KV 导致短请求饿死；
- 降低 preemption 后全量 recompute；
- 改善 p95/p99 ITL 和 queueing delay；
- 提高 device KV active pages 利用率。

需要测的 counter：

- retract count / 1K requests；
- retract victim token length 分布；
- writeback bytes / retract；
- recovery loadback bytes；
- recovery latency；
- retracted duration；
- retraction 后 short request admission success；
- recomputed prefill tokens。

收益估计：

```text
KV memory 充足：接近 0。
KV pressure 高但 prefix reuse 弱：小到中等，主要降低失败/排队。
KV pressure 高且 repeated prefix 强：p95/p99 可能有 10-25% 改善。
```

## 8. 优化七：cache movement 与 forward 在同一个 ExecutionPlan 中调度，并在 TP ranks 上一致提交

### 8.1 具体实现

`Scheduler::NextExecutionPlan()` 每轮不是只返回 forward batch，而是同时返回：

- `FlatForwardOperation`；
- `FlatWriteBackOperation`；
- `FlatLoadBackOperation`。

源码：`tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:418-458`。

流程：

1. 先为 `Draining` 请求生成 writeback ops；
2. 清理 `Finished` 请求；
3. 过滤掉 `Draining / Prefetching / Retracting / WritingBack`；
4. 对 candidates 调 `newForwardOperation()`；
5. 把 forward ops 和 cache ops 合进 plan。

Python `EventLoop` 每轮会：

- 先 `_commit_cache_results()`，把已完成 cache event 回灌给 scheduler；
- 调 `scheduler.next_execution_plan()`；
- `_submit_cache_ops(execution_plan)`；
- 再 dispatch forward。

源码：`event_loop.py:1039-1053`。

cache event 不是单 rank 随便提交。`_commit_cache_results()` 会：

- 从 `MemoryExecutor.poll_results()` 取本地结果；
- 把结果转成 payload；
- TP > 1 时用 `all_gather_object` 聚合各 rank payload；
- 只 pop common cache events；
- 统一 `scheduler.advance(ec)`。

源码：`event_loop.py:419-453`，`event_loop.py:472-502`。

### 8.2 为什么对 agent KV 有意义

Agent 场景下，cache movement 与 forward 的先后顺序很容易出 correctness bug：

- 本轮新请求要写一个 device page；
- 上轮 retract/writeback 还在读同一个 page；
- TP rank 之间 cache op 完成顺序不同；
- scheduler mirror 发生分叉；
- 下一轮 prefix match 使用了某个 rank 尚未完成的 page。

TokenSpeed 在 `_setup_layerwise_loadback()` 里有一个非常直接的 fence：

```text
writeback 需要和本轮 set_kv_buffer 排序；
scheduler 可能在同一 iter 把 freed-but-not-yet-written-back slot
重新分配给新的 prefill/decode；
因此 execution_stream 要 wait host write_stream。
```

源码：`event_loop.py:630-642`。

这说明它在处理一个很具体的 KV correctness 边界：**async writeback 读旧 page 与新 forward 写同一 page 的顺序**。

### 8.3 复制难度

这个优化不是简单“加异步 copy”。必须有：

- scheduler 输出 cache ops；
- MemoryExecutor 异步执行；
- cache result event 回灌；
- TP ranks common event commit；
- page ownership 在 transfer 期间被 pin；
- write stream 与 execution stream 的 fence。

这属于 runtime protocol 级复制。

## 9. 优化八：paged cache group 支持非标准 KV / latent cache layout 的独立 admission

### 9.1 具体实现

TokenSpeed 不只管理主 KV page，还支持从 `token_to_kv_pool` 暴露 `paged_cache_group_specs`，转成 scheduler 的 paged cache group config：

- `scheduler_utils.py:102-129`：`pool_to_paged_cache_groups()` 支持 `full_history` 和 `sliding_window` retention。
- `event_loop.py:278-300`：构造 scheduler config 时传入 `paged_cache_groups`。
- `scheduler.cpp:257-383`：对每个 group 做 admission、sliding-window release、Acquire、并把 group page ids 写入 forward op。
- `model_executor.py:939-951`：forward 时构造 `paged_cache_block_tables` 和 base offsets，传给 `forward_step`。

### 9.2 Agent 场景为什么相关

V4/Kimi 类模型不一定只有传统 MHA KV。latent-KV、compressed KV、sliding-window cache、额外 index/cache table 都可能有不同 retention 规则。

如果 scheduler 只知道“主 KV page”，就可能出现：

```text
主 KV 够，但 latent/cache group 不够 -> forward 失败
或者 sliding-window 本可释放，却仍按 full-history 占用
```

TokenSpeed 把这些额外 cache group 也放进 scheduler admission，至少说明它的 KV 管理不是单一 block table，而是能表达多 cache resource 的联合 admission。

### 9.3 需要进一步确认

这块要避免过度表述：

- 已确认代码存在 paged cache group admission 和 forward op 物化；
- 需要继续确认 DeepSeek V4 / Kimi 具体哪些 cache layout 实际使用了这些 group；
- 对 Ascend 迁移时，这可能是 latent-KV / compressed cache 的重要接口，但不能直接估收益。

## 10. 优化九：DP dispatch 可以按 cache usage，而不是只按请求数

### 10.1 具体实现

在 attention-DP 模式下，TokenSpeed 的 `DataParallelController` 支持三种策略：

- `ROUND_ROBIN`；
- `SHORTEST_QUEUE`；
- `MINIMUM_CACHE_USAGE`。

源码：`python/tokenspeed/runtime/engine/data_parallel_controller.py:56-77`。

`MINIMUM_CACHE_USAGE` 使用每个 worker 返回的 `num_pages` 作为 load metric。worker 侧 `_get_load()` 会从 C++ scheduler 查询：

```python
available = self.scheduler.available_kv_pages()
num_total_pages = self.max_total_num_tokens // block_size
num_used_pages = num_total_pages - available
```

源码：`event_loop.py:838-853`。

### 10.2 为什么这是 KV 管理的一部分

Agent 多轮 session 的 cost 不等于 request count。一个 DP rank 上 4 个短请求，可能比另一个 rank 上 1 个超长 session 更轻。

如果 DP dispatch 只看 request count，会出现：

```text
rank0: 少量长 session，占满 KV
rank1: 多个短 session，KV 仍宽松

request-count 视角：rank0 不忙
KV 视角：rank0 很危险
```

TokenSpeed 至少把 cache usage 暴露给 admission 层，使新 session 更可能落到 KV footprint 更低的 rank 上。

### 10.3 性能收益路径

主要看：

- pages per DP rank variance；
- retraction count per DP rank；
- waiting queue per rank；
- p95 queueing delay；
- zero-token rank count；
- DP rank tokens variance。

如果 agent sessions 很长且 DP rank 间 sticky，cache-aware dispatch 可能明显降低局部 KV pressure；如果所有请求长度接近，收益会很小。

## 11. 优化十：abort / grammar admission 直接影响 KV slot 和无效 forward

这不是传统 KV cache 优化，但在 agent 场景里会直接影响 KV 资源占用。

### 11.1 具体实现

`RequestHandler` 收到 `AbortReq` 后把 rid 放到 `abort_rids`。源码：`request_handler.py:145-181`。

`EventLoop._process_new_requests()` 会：

- `output_processor.mark_abort(rid)`；
- `grammar_manager.mark_abort(rid)`；
- 对 grammar 编译失败或已 abort 的请求，不进入 scheduler，直接 `publish_finished_at_admission()`。

源码：`event_loop.py:699-782`。

`OutputProcessor.mark_abort()` 的注释很关键：如果 abort 不直接 materialize `finished_reason`，scheduler 会继续跑到 max_tokens/EOS，浪费 forward steps，并 latch `--max-num-seqs` slot。

源码：`generation_output_processor.py:313-323`。

postprocess 时，已 abort 请求会：

- 追加 ExtendResult；
- 追加 FinishEvent；
- 从 `rid_to_state` 删除；
- 不再输出给 detokenizer。

源码：`generation_output_processor.py:578-584`。

### 11.2 Agent 场景为什么重要

Agent runtime 中用户取消、tool timeout、grammar invalid、parallel branch 失败都很常见。每一个未及时清理的请求都会继续占：

- req pool slot；
- request-local KV pages；
- output processor state；
- decode batch slot；
- 可能还会触发后续 reserve。

因此 abort path 是 KV 管理的外围保障：它保证无效请求尽快向 C++ scheduler 发送 Finish/Abort 语义，释放资源，避免污染 p95。

## 12. 性能提升应该怎么拆账

不能把这些优化简单相加。更合理的拆账方式是按瓶颈迁移：

| 优化 | 主要移动的 counter | 受益 workload | 估计收益边界 |
|---|---|---|---|
| device prefix reuse / finish insert | cached token ratio、prefill tokens saved、TTFT | 多轮 repeated prefix | 5-20%，取决于重复 prefix 占比 |
| host/device loadback/writeback | recompute tokens、loadback bytes、cache transfer p95 | KV 可落到 host 且 reuse 强 | V4 当前禁用 kvstore，短期不能记入 |
| tail page + dynamic reserve | tail page waste、admission failure、decode p95 | 短 decode / spec decode | 2-8%，更偏 p95 稳定性 |
| retract / recovery | retraction count、recovery latency、queue p95 | KV pressure + 长 session 混排 | 10-25% p95/p99，内存充足时接近 0 |
| ExecutionPlan cache+forward | GPU gap、cache op wait、TP scheduler divergence | async cache movement + TP | 5-15% tail，取决于 transfer 频率 |
| paged cache groups | non-main cache allocation failure、sliding release | latent/compressed cache | 待模型路径确认 |
| cache-aware DP dispatch | pages/rank variance、rank queue p95 | 多 DP 长 session | 5-15% tail，取决于 DP skew |
| abort / grammar admission | wasted decode steps、slot cleanup latency | tool timeout / cancellation 多 | 低到中，但对 p95 有价值 |

### 12.1 架构竞争力判断应关注的性能信号

正式报告不需要展开实验计划，但需要说明这些 runtime 信号能支撑竞争力判断：

```text
prefix/cache
  - device prefix hit pages / tokens
  - host prefix hit pages / tokens
  - cached token ratio
  - prefill tokens saved

page pressure
  - available_kv_pages
  - active_kv_pages
  - cached pages
  - tail page waste
  - admission failure / retry

cache movement
  - loadback op count / bytes / latency
  - writeback op count / bytes / latency
  - cache op wait p95
  - TP common cache event lag

retract/recovery
  - retract count
  - victim token length
  - retracted duration
  - recovery loadback bytes
  - recovery latency
  - recompute tokens after recovery

agent runtime
  - wasted decode steps after abort
  - req_pool occupancy
  - per-rank pages variance
  - p50/p95/p99 TTFT
  - p50/p95/p99 ITL
```

## 13. 技术解读组织建议

这部分不应该写成“TokenSpeed 支持 prefix cache / retract / kvstore”的列表，而应该按机制链条讲：

1. **Agent KV 问题不是长上下文，而是多轮短 decode + KV pressure**
   - 讲 repeated prefix、short decode、request churn、DP cache skew。

2. **TokenSpeed 的关键抽象：request state = KV owner**
   - 用状态机图讲 `Submitted -> Prefilling -> Decoding -> Draining/WritingBack -> Finished` 和 `Retracting/Retracted -> Decoding`。

3. **prefix reuse 如何变成 execution plan**
   - device/host depth、loadback diff、ForwardOp + CacheOp。

4. **decode 期间如何避免 page pressure**
   - local tail page、dynamic reserve、output feedback。

5. **KV pressure 下如何 retract/recover**
   - longest victim、device->host writeback、Retracted state、loadback recovery、hist_token_len。

6. **为什么 vLLM 不容易小 patch 复制**
   - 单点能力可复制，完整 ownership protocol 是 scheduler/KV/event loop 架构级复制。

7. **收益如何归因**
   - 说明 10-25% 这类区间只应被理解为 repeated-prefix + memory-pressure 场景下的可能改善，不是所有场景平均提升。

## 14. 当前结论

TokenSpeed 为 agent 场景做的 KV 管理优化，可以概括为四句话：

```text
第一，KV page 不只是 cache block，而是 request FSM 状态持有的资源。

第二，prefix reuse 不只是 hash 命中，而是 device/host depth、
loadback diff、page admission 和 ForwardOp/CacheOp 的联合计划。

第三，decode KV append 被细化为 tail page + dynamic reserve，
适合短 decode / spec decode 的小步迭代。

第四，KV pressure 下通过 retract/recovery 把长 session 暂时转成
host-backed recoverable state，而不是简单 preempt 后重算。
```

对 vLLM/vLLM-Ascend 的复制判断也应更克制：

- prefix cache、block hash、admission heuristic 是可复制的；
- cache-aware DP dispatch 是中等复杂度；
- host/device loadback、retract/recovery、async transfer pinning 是中高复杂度；
- 完整 request FSM + KV ownership + event-loop cache/forward protocol 是架构级复制。

对适配判断的含义：

- 短期不要把 DeepSeek V4 hierarchical kvstore 收益算进去，因为当前 TokenSpeed V4 path 禁用 kvstore；
- 优先判断 device KV ownership、prefix safe reuse、tail page / reserve、retract/recovery 语义、DP cache-aware dispatch 是否形成 vLLM 不易小 patch 复制的 runtime protocol；
- 如果 repeated-prefix + KV-pressure agent trace 下，TokenSpeed 相比 vLLM-Ascend 改善 p95/p99 且不靠单个 MLA kernel，那么 Scheduler/KV ownership 才能成立为核心护城河候选。
