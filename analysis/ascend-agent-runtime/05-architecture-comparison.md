# TokenSpeed / vLLM / TensorRT-LLM 架构与实现对照 v0.6

更新时间：2026-05-24

本文替换上一版“竞争力候选”写法。上一版的问题是先把 TokenSpeed 放在“可能更特别”的框架里，再去解释哪些机制支撑这个判断，容易变成带结论找证据。

这一版只做客观对照：

```text
同一个系统问题：
  1. agent workload 下 CPU 控制面高负载
  2. agent workload 下多轮 KV 管理

分别看：
  TokenSpeed 当前如何实现？
  vLLM V1 当前如何实现？
  TensorRT-LLM 当前如何实现？

再用同一条带 tool call 的 agent 请求贯穿：
  Round 1: messages + tools -> model emits tool_call
  Tool wait: external tool executes
  Round 2: messages + tool result -> model continues answer
```

本文不判断 TokenSpeed 是否明显更强，只列出架构差异、实现路径、边界条件和仍需验证的问题。

## 关于竞争力结论的说明

本文应与 [TokenSpeed 竞争力验证框架](08-competitiveness-validation-framework.md) 一起阅读。

本文的目的不是声称 TokenSpeed 已经优于 vLLM 或 TensorRT-LLM，而是识别哪些架构差异在得到源码证据、profiling counters 和可复现 benchmark 结果支撑后，才可能变成真实优势。

在本文中：

- “潜在优势”表示架构上可能带来收益，但仍缺 benchmark 证据。
- “已验证优势”只能用于受控 workload 下已经证明收益的场景。
- tool-call parsing 这类 protocol-layer feature 不应被视为 TokenSpeed engine 优势，除非 engine 本身观测并使用该状态。
- 必须区分 SMG gateway capabilities 和 TokenSpeed engine capabilities。

## 0. 证据边界

| 框架 | 本文证据来源 | 本文不做的事 |
|---|---|---|
| TokenSpeed + SMG | 本仓源码、`analysis/ascend-agent-runtime` 前序源码分析、SMG 拆解文档 | 不把未读代码推断成已实现能力 |
| vLLM / vLLM-Ascend | vLLM 官方架构、V1 guide、prefix caching、hybrid KV、disaggregated prefill 文档 | 不重新深挖 vLLM 源码 |
| TensorRT-LLM | NVIDIA 官方 architecture、Executor API、scheduler、KV cache system 文档 | 不推断非 NVIDIA 平台表现 |

需要特别保留的 TokenSpeed 边界：

- tool parser / MCP tool loop / OpenAI protocol 主要在 SMG gateway，不在 TokenSpeed C++ Scheduler。
- TokenSpeed engine 里没有看到 tool-call-specific DDR KV offload/resume 逻辑。
- DeepSeek V4 当前 baseline 对 hierarchical KVStore 有明确限制，不能把 host/L2/L3 offload 当作 V4 已落地收益。
- DeepSeek V4 当前更像手写模型逻辑 + `CommManager` + 自定义 attention/MoE，不应说已完全走 generic placement compiler。

## 1. 三个框架的整体架构图

### 1.1 TokenSpeed：SMG gateway + Python runtime + 嵌入式 C++ Scheduler/KV + GPU execution plane

```mermaid
flowchart LR
  U["Client / Agent App"]

  subgraph SMG["SMG Gateway Process"]
    P0["OpenAI protocol"]
    CT["chat template"]
    TC["tokenizer L0/L1 cache"]
    TP["tool parser / reasoning parser"]
    MCP["MCP tool loop"]
    RT["worker routing"]
    P0 --> CT --> TC --> TP --> RT
    MCP --> P0
  end

  subgraph MAIN["Main Python Process"]
    ENG["Engine"]
    ALLM["AsyncLLM"]
    IP["InputProcessor / tokenizer"]
    FRS["front-end ReqState / collector"]
    OUT["AsyncLLM OutputProcessor / detokenize"]
    ENG --> ALLM --> IP
    ALLM --> FRS
    OUT --> FRS
  end

  subgraph DPC["Optional DataParallelController"]
    DPLB["DP load balance / cache usage signal"]
  end

  subgraph WORKER["Scheduler Worker Python Process: one per rank"]
    RH["RequestHandler<br/>TP rank0 recv + TP broadcast"]
    EL["EventLoop<br/>admission + overlap loop"]
    MEM["MemoryExecutor<br/>cache op when enabled"]
    MX["ModelExecutor<br/>req_to_page + InputBuffers"]
    GOP["Generation OutputProcessor<br/>Extend / Finish / Reserve"]
    RH --> EL
    EL --> MEM
    EL --> MX
    MX --> GOP
    GOP --> EL
  end

  subgraph CXX["Embedded C++ Scheduler"]
    SCH["Scheduler"]
    FSM["Request FSM"]
    KVP["KVPrefixCache / HybridPrefixCache"]
    PA["PageAllocator / ReqPoolAllocator"]
    EP["ExecutionPlan<br/>ForwardOp + CacheOp"]
    SCH --> FSM
    SCH --> KVP
    SCH --> PA
    SCH --> EP
  end

  subgraph GPU["GPU Execution Plane"]
    TKV["token_to_kv_pool<br/>physical KV tensors"]
    RTP["req_to_page block table"]
    IB["InputBuffers"]
    FC["ForwardContext"]
    MODEL["ModelRunner / attention / dense / MoE / sampling"]
    TKV --> MODEL
    RTP --> MODEL
    IB --> MODEL
    FC --> MODEL
  end

  U --> SMG
  SMG -->|"tokenized request / sampling constraint"| MAIN
  MAIN -.->|"if attention-DP"| DPC
  MAIN -->|"ZMQ request"| RH
  DPC --> RH
  EL -->|"submit / advance"| SCH
  EP -->|"CacheOp"| MEM
  EP -->|"ForwardOp"| MX
  MX --> GPU
  GOP -->|"BatchTokenIDOut"| OUT
  OUT --> SMG
```

**源码对应关系：**

| 图中组件 | TokenSpeed / SMG 证据 |
|---|---|
| SMG gateway | `python/tokenspeed/cli/serve_smg.py`、`python/tokenspeed/cli/_proc.py`、`analysis/ascend-agent-runtime/07-smg-gateway-runtime.md` |
| AsyncLLM / InputProcessor | `python/tokenspeed/runtime/engine/async_llm.py`、`input_processor.py` |
| RequestHandler / EventLoop | `python/tokenspeed/runtime/engine/request_handler.py`、`event_loop.py` |
| C++ Scheduler / FSM | `tokenspeed-scheduler/csrc/scheduler/*`、`tokenspeed-scheduler/csrc/fsm/*` |
| KV ownership | `tokenspeed-scheduler/csrc/resource/allocator/*`、`kv_prefix_cache/*`、`hybrid_prefix_cache/*` |
| GPU materialization | `python/tokenspeed/runtime/execution/model_executor.py`、`input_buffer.py`、`runtime_states.py` |
| model/parallel path | `python/tokenspeed/runtime/models/deepseek_v4.py`、`runtime/distributed/*` |

### 1.2 vLLM V1：API Server Process + EngineCore + GPU Worker Processes

```mermaid
flowchart LR
  U["Client / OpenAI-compatible API"]

  subgraph API["API Server Process"]
    HTTP["HTTP / OpenAI protocol"]
    INP["input processing<br/>tokenization / multimodal loading"]
    STREAM["detokenization / streaming"]
    HTTP --> INP
    STREAM --> HTTP
  end

  subgraph EC["Engine Core Process"]
    LOOP["EngineCore busy loop"]
    SCHED["Unified Scheduler<br/>token budget"]
    KVM["KVCacheManager"]
    OUTP["output handling"]
    LOOP --> SCHED
    SCHED --> KVM
    LOOP --> OUTP
  end

  subgraph DP["Optional DP Coordinator"]
    DPLB["DP load balancing<br/>sync forward for MoE"]
  end

  subgraph GW["GPU Worker Processes"]
    WRK["worker per GPU"]
    MR["ModelRunner"]
    KVC["GPU KV cache blocks"]
    CUDA["CUDA graph / kernels"]
    WRK --> MR
    KVC --> MR
    MR --> CUDA
  end

  U --> API
  API --> EC
  EC -.-> DP
  DP -.-> GW
  EC --> GW
  GW --> EC
  EC --> API
```

**官方文档事实：**

- vLLM V1 process architecture 包括 API Server Process、Engine Core Process、GPU Worker Processes；DP 时还有 DP Coordinator。
- 官方文档的 process count summary 写明：API Server 处理 HTTP 和 input processing；Engine Core 负责 scheduler 和 KV cache management；GPU Worker 执行 model forward；DP Coordinator 负责 DP rank load balancing。
- vLLM V1 unified scheduler 用 `{request_id: num_tokens}` 形式的 token budget 统一 prompt/output token scheduling，服务 chunked prefill、prefix caching、spec decode。

### 1.3 TensorRT-LLM：LLM/Executor API + PyExecutor/C++ runtime + TensorRT engine

```mermaid
flowchart LR
  U["Client / LLM API / Triton / service"]

  subgraph API["TensorRT-LLM API Layer"]
    LLM["LLM class / Executor API"]
    ENQ["enqueueRequest(s)"]
    RESP["awaitResponses / streaming result"]
    LLM --> ENQ
    RESP --> LLM
  end

  subgraph EXEC["Executor / PyExecutor background loop"]
    Q["request queue"]
    SCH["Scheduler"]
    KVM["KVCacheManager"]
    ME["ModelEngine"]
    SMP["Sampler"]
    OH["output handling"]
    Q --> SCH
    SCH --> KVM
    SCH --> ME
    ME --> SMP
    SMP --> OH
  end

  subgraph OPT["Runtime Optimizations"]
    IFB["in-flight batching"]
    CG["CUDA Graph"]
    OS["Overlap Scheduler"]
    SPEC["speculative decoding"]
  end

  subgraph GPU["NVIDIA GPU / TensorRT Engine"]
    TRT["TensorRT engine"]
    KV["paged KV cache"]
    TRT --> KV
  end

  U --> API
  ENQ --> Q
  EXEC --> OPT
  ME --> TRT
  OH --> RESP
```

**官方文档事实：**

- TensorRT-LLM architecture 文档把 `LLM` class 描述为核心入口，并说明内部会创建每个 rank 的 `PyExecutor(Worker)` background loop。
- 该 loop 关键组件包括 Scheduler、KVCacheManager、ModelEngine、Sampler。
- Executor API 是 high-level C++ API，支持 async request execution 和 in-flight batching。
- Runtime optimization 包括 CUDA Graph 和 Overlap Scheduler；官方文档描述 Overlap Scheduler 会在 CPU 处理上一轮结果时启动下一轮 GPU work。

### 1.4 整体架构横向对照

| 维度 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| user-facing gateway | SMG 独立 gateway，处理 protocol/template/parser/tool loop/routing | API Server Process，处理 HTTP/input/output | LLM API / Executor / Triton serving |
| 调度主循环 | Python EventLoop 调 C++ Scheduler | EngineCore busy loop | Executor / PyExecutor background loop |
| scheduler 形态 | 嵌入 worker 的 C++ request FSM + ExecutionPlan | V1 unified scheduler，token budget 表达 | Scheduler 与 KVCacheManager 在 executor loop 内协同 |
| KV 管理边界 | C++ logical page ownership + Python materialization + GPU physical KV | KVCacheManager / BlockPool / KVCacheBlock | KVCacheManager / KVCacheConfig / paged KV / offload/events |
| GPU 执行 | Python ModelExecutor + ModelRunner + custom backends | GPU workers + ModelRunner + kernels/CUDA graph | TensorRT engine / ModelEngine + CUDA Graph |
| agent/tool 边界 | parser/tool loop 在 SMG，engine 不感知 tool object | API/tool examples 支持，engine 侧主要处理 tokenized request | serving/API 层支持 OpenAI examples，runtime 侧以 Executor request 为核心 |
| 需要注意 | DeepSeek V4 hierarchical KVStore 当前不可作为已落地收益 | vLLM V1 已专门重构 CPU/KV 路径 | 强绑定 NVIDIA runtime 和 TensorRT engine |

### 1.5 同一条 agent 请求的系统边界

后续 CPU / KV 对照都基于同一条请求，不再抽象讨论：

```text
Round 1:
  messages = [system, user]
  tools = [get_weather(city, date)]
  model output = tool_call(get_weather, {"city": "北京", "date": "明天"})

Tool wait:
  agent / gateway / app 执行 get_weather

Round 2:
  messages = [system, user, assistant(tool_call), tool(result)]
  tools = same tool schema or reduced tool schema
  model output = final answer
```

三个框架的共同点是：模型执行引擎通常不直接执行外部工具。真正进入 engine/executor 的通常是渲染后的 prompt、token ids、sampling 参数、grammar/constraint 或 executor request。差别在于 tool/template/parser/routing 这些 CPU 控制面放在什么层，以及 Round 2 是否能命中 Round 1/历史 prefix 的 KV。

```mermaid
flowchart LR
  A["Agent App<br/>messages + tools"]
  G["Gateway / API layer<br/>template + parser + routing"]
  E["Engine / Executor<br/>scheduler + KV manager"]
  M["GPU model forward"]
  P["Parser / output adapter<br/>tool_call extraction"]
  T["External tool runtime"]
  R2["Round 2 request<br/>history + tool result"]

  A --> G --> E --> M --> P --> T --> R2 --> G
```

| 阶段 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| tool schema/template | SMG gateway / tokenizer wrapper | API server / serving layer | LLM API / serving layer |
| parser/tool loop | SMG tool parser / MCP loop | API/server/tool calling layer，engine 文档不把它作为 scheduler 核心 | serving/API 层，Executor 侧以 request 为核心 |
| engine 入口 | tokenized request + sampling/grammar constraint | processed request 进入 EngineCore | enqueueRequest(s) 进入 Executor |
| Round 2 复用前提 | prompt 稳定 + worker routing + engine prefix cache | prompt 稳定 + APC / KV manager / routing | prompt 稳定 + KV reuse / optional cache-aware routing/events |
| tool wait 期间 KV | 未看到 tool-call-specific DDR offload/resume | 官方 V1 guide 显示 GPU<>CPU KV swapping removed；可通过其他 KV connector/disagg 机制讨论 | `host_cache_size` 等 KV offload 能力存在，但具体 serving policy 由部署决定 |

## 2. 相同问题一：Agent CPU 高负载如何处理

### 2.1 问题本身

Agent workload 的 CPU 高负载不只是“HTTP 请求多”。它通常来自：

```mermaid
flowchart LR
  MSG["messages / tools / tool_choice"]
  TMP["chat template"]
  TOK["tokenization"]
  ADM["admission / grammar compile"]
  SCH["per-step scheduling"]
  OUT["output processing<br/>detok / stop / parser"]
  LOOP["tool loop / next request"]
  GPU["GPU forward"]

  MSG --> TMP --> TOK --> ADM --> SCH --> GPU --> OUT --> LOOP --> MSG
```

decode step 越短，GPU forward 越快，CPU 侧的固定工作越容易变成 p95 ITL 或 GPU idle gap。比较三个框架时，应该看 CPU 控制面被拆到哪里、哪些动作和 GPU forward overlap、哪些动作仍在请求关键路径上。

### 2.2 TokenSpeed 的处理路径

```mermaid
sequenceDiagram
  participant SMG as SMG Gateway
  participant A as AsyncLLM/InputProcessor
  participant E as EventLoop
  participant S as C++ Scheduler
  participant M as ModelExecutor/GPU
  participant O as Generation OutputProcessor

  SMG->>SMG: protocol / template / tokenizer cache / tool parser
  SMG->>A: tokenized request + sampling/grammar constraint
  A->>E: ZMQ request
  E->>E: grammar admission / abort handling / cache query
  E->>S: submit requests
  loop each scheduler step
    E->>S: NextExecutionPlan()
    S-->>E: ForwardOp + CacheOp
    E->>M: dispatch current forward
    M-->>O: async output copy event
    O-->>E: ExtendResult / Finish / Reserve event
    E->>S: advance(previous output/cache events)
  end
  O-->>A: BatchTokenIDOut
  A-->>SMG: detokenized stream / text
  SMG->>SMG: tool/reasoning parse
```

**实现事实：**

- SMG 吃掉一部分 agent-facing CPU 工作：OpenAI protocol、chat template、tokenizer cache、tool parser、MCP loop、routing。
- TokenSpeed engine 的 `GenerateReqInput` 不直接携带原始 tools；进入 engine 后主要是 token ids / text、sampling params、grammar/structural constraint。
- `EventLoop` 把 cache result commit、scheduler plan、cache op submit、forward dispatch、output commit、scheduler.advance 拆开。
- overlap loop 的核心是当前 forward 先 dispatch，再处理上一轮结果，让 CPU postprocess/advance 有机会与 GPU forward overlap。
- C++ Scheduler 负责 request/KV 决策，但模型 forward、InputBuffers、output processing、部分 distributed metadata 仍在 Python worker 里。

**在同一问题下的客观表达：**

```text
TokenSpeed 的 CPU 控制面被分成两段：
  SMG 处理 user-facing agent 协议；
  engine worker 用 EventLoop + C++ Scheduler 处理 token-level scheduling/KV。

需要继续验证的是：
  在真实 short-decode agent workload 下，
  SMG parser/tool loop、AsyncLLM、EventLoop、C++ Scheduler、OutputProcessor
  哪一段是主要 CPU 时间来源。
```

### 2.3 vLLM V1 的处理路径

```mermaid
sequenceDiagram
  participant API as API Server Process
  participant EC as EngineCore Process
  participant S as Unified Scheduler
  participant K as KVCacheManager
  participant W as GPU Worker
  participant OUT as API output/streaming

  API->>API: HTTP / input processing / tokenization
  API->>EC: request to EngineCore
  loop each engine step
    EC->>S: allocate token budget
    S->>K: get_computed_blocks / allocate_slots
    EC->>W: dispatch model forward
    W-->>EC: output tokens
    EC-->>API: output packets
  end
  API->>OUT: detokenization / streaming
```

**官方文档事实：**

- vLLM V1 process architecture 将 API server、Engine Core、GPU workers 拆成独立进程。
- API Server Process 处理 HTTP requests 和 input processing；Engine Core Process 管 scheduler 和 KV cache management；GPU Worker 执行 model forward。
- V1 unified scheduler 用 token budget 同时处理 prompt/output tokens，支持 chunked prefill、prefix caching、speculative decoding。
- V1 guide 状态表显示 prefix caching、chunked prefill、FP8 KV cache、spec decode 为 functional；GPU <> CPU KV cache swapping 被移除。

**在同一问题下的客观表达：**

```text
vLLM V1 的 CPU 控制面设计重点是进程拆分和 EngineCore busy loop：
  API/input/output 与 scheduler/model executor 分离；
  scheduler 用统一 token budget 降低 prefill/decode 特例分支；
  GPU worker 承担 forward。

本文不判断其优劣，只记录它不是“单进程 Python scheduler”。
```

### 2.4 TensorRT-LLM 的处理路径

```mermaid
sequenceDiagram
  participant C as Client
  participant EX as Executor / PyExecutor
  participant S as Scheduler
  participant K as KVCacheManager
  participant G as ModelEngine / TensorRT Engine
  participant P as Sampler / Output

  C->>EX: enqueueRequest(s)
  loop background loop
    EX->>S: fetch + schedule active requests
    S->>K: reserve KV resources
    EX->>G: launch GPU work for step n
    G-->>P: logits / tokens
    par overlap
      EX->>G: launch GPU work for step n+1
    and
      P->>EX: process previous batch on CPU
    end
  end
  EX-->>C: awaitResponses / streaming
```

**官方文档事实：**

- Executor API 支持异步请求执行和 in-flight batching。
- PyExecutor background loop 每轮做 request fetching、scheduling、KV resource preparation、model execution、output handling。
- CUDA Graph 用于降低 CPU-side kernel launch overhead。
- Overlap Scheduler 的策略是 CPU 处理当前/上一轮输出时，GPU 已经启动下一步 work；官方文档说默认开启。

**在同一问题下的客观表达：**

```text
TensorRT-LLM 的 CPU 控制面设计重点是 executor/runtime-first：
  request execution 在 Executor/PyExecutor loop 中；
  GPU launch overhead 通过 CUDA Graph 处理；
  CPU result processing 通过 Overlap Scheduler 与下一步 GPU work 重叠。

本文不把该能力迁移到 Ascend 做推断，因为它依赖 NVIDIA runtime/TensorRT engine。
```

### 2.5 CPU 高负载实现对照图

```mermaid
flowchart TB
  Q["同一问题：短 decode + tool loop 下 CPU 控制面是否拖住 GPU"]

  subgraph TS["TokenSpeed"]
    TS1["SMG: protocol/template/tokenizer/parser/routing"]
    TS2["AsyncLLM: request state + detok"]
    TS3["EventLoop: cache/forward/output/advance loop"]
    TS4["C++ Scheduler: request/KV decision"]
    TS5["GPU forward"]
    TS1 --> TS2 --> TS3 --> TS4 --> TS3 --> TS5
  end

  subgraph VL["vLLM V1"]
    VL1["API Server: HTTP/input/output"]
    VL2["EngineCore busy loop"]
    VL3["Unified Scheduler: token budget"]
    VL4["KVCacheManager"]
    VL5["GPU workers"]
    VL1 --> VL2 --> VL3 --> VL4 --> VL5
  end

  subgraph TRT["TensorRT-LLM"]
    TR1["LLM/Executor API"]
    TR2["PyExecutor/C++ loop"]
    TR3["Scheduler + KVCacheManager"]
    TR4["CUDA Graph"]
    TR5["Overlap Scheduler"]
    TR6["TensorRT engine"]
    TR1 --> TR2 --> TR3 --> TR4 --> TR6
    TR5 --> TR6
  end

  Q --> TS
  Q --> VL
  Q --> TRT
```

| 对照点 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| agent protocol / tool parser | SMG gateway | API server/examples/tool calling path | serving/API layer examples |
| tokenization / detokenization | SMG + AsyncLLM output path | API Server Process | LLM class handles pre/post; serving layer depends deployment |
| scheduler CPU path | C++ Scheduler embedded in Python worker | EngineCore Process unified scheduler | Scheduler inside Executor/PyExecutor loop |
| CPU/GPU overlap | EventLoop overlap between forward and previous output/advance | multiprocess API/EngineCore split; persistent batch/chunked prefill per docs | CUDA Graph + Overlap Scheduler |
| still in CPU critical path | parser/tool loop, IPC, EventLoop, output processor, Python materialization | API input/output, EngineCore scheduling, KV manager | executor scheduling, output handling, runtime callbacks |
| unresolved for this report | real CPU time attribution per component | no vLLM source-level profiling here | NVIDIA-specific runtime assumptions |

### 2.6 同一条 agent 请求的 CPU 控制面阶段对照

这一节把上面的图拆成执行阶段。目标不是判断谁更快，而是明确每个框架在相同阶段把 CPU 工作放在哪里。

```mermaid
sequenceDiagram
  participant APP as Agent App
  participant G as Gateway / API
  participant E as Engine / Executor
  participant S as Scheduler / KV Manager
  participant GPU as GPU Forward
  participant P as Parser / Output
  participant TOOL as External Tool

  APP->>G: Round 1 messages + tools
  G->>G: template / tokenize / parser constraint
  G->>E: engine request
  E->>S: admission + scheduling
  S->>GPU: prefill + short decode
  GPU-->>P: output tokens
  P->>P: stop / detok / tool_call parse
  P->>TOOL: execute tool
  TOOL-->>APP: tool result
  APP->>G: Round 2 history + tool result
  G->>E: second engine request
  E->>S: prefix lookup + scheduling
  S->>GPU: mostly cached prefill + short decode
```

| CPU 阶段 | TokenSpeed 当前路径 | vLLM V1 当前路径 | TensorRT-LLM 当前路径 |
|---|---|---|---|
| Round 1 HTTP / protocol | SMG gateway | API Server Process | LLM API / serving layer |
| chat template / tools rendering | SMG / tokenizer wrapper；tools 变成 prompt 或 constraint | API/input processing 层；EngineCore 文档关注 processed request | LLM API / serving 层，Executor request 不等于原始 OpenAI messages |
| tokenizer cache / tokenization | SMG tokenizer L0/L1 + AsyncLLM/InputProcessor 路径 | API Server Process 处理 input processing / tokenization | LLM class / serving layer处理，具体依部署 |
| grammar / structured constraint | SMG 生成约束；engine admission 的 GrammarManager 处理 ready/timeout/abort | guided decoding / structured output 属 API/engine 功能，但本文未深挖源码 | 支持取决于 LLM API / serving 集成；本文只看 Executor 文档 |
| admission | EventLoop `_process_new_requests()`，处理 grammar、abort、L3 query、PD hints | EngineCore scheduler 接收 request 后统一调度 | Executor/PyExecutor loop fetch requests |
| per-step schedule | C++ Scheduler `NextExecutionPlan()` 产出 `ForwardOp + CacheOp` | Unified Scheduler 分配 token budget，并和 KVCacheManager 交互 | Scheduler/CapacityScheduler 与 KVCacheManager 交互 |
| GPU launch 固定开销 | Python ModelExecutor materialization + CUDA graph wrapper path；仍有 Python runtime | GPU Worker / ModelRunner / CUDA graph 相关能力 | CUDA Graph 是官方 runtime optimization |
| output / detok | Generation OutputProcessor -> AsyncLLM OutputProcessor inline detok -> SMG parser | EngineCore/API Server output packets、detokenization/streaming | Sampler/output handling -> awaitResponses / streaming |
| tool_call parse | SMG parser `parse_complete` / `parse_incremental` | API/server tool calling layer；不属于 V1 scheduler/KV 文档核心 | serving/API 层；Executor 不执行外部 tool |
| tool wait | 外部 tool/MCP loop；engine request 已 finish | 外部 app/server loop；engine request 已 finish | 外部 app/server loop；Executor request 已 finish |
| Round 2 request build | SMG 重新渲染 history + tool result；routing 可影响 worker locality | API server 重新处理 history + tool result | serving layer 重新 enqueue request |

从这个阶段表可以得到几个中性事实：

1. **CPU 高负载不是单个 scheduler 问题。**
   template、tokenization、parser、detok、tool loop、per-step scheduling 都可能贡献 CPU 时间。

2. **TokenSpeed 的 tool parser 不减轻 engine scheduler 本身的工作。**
   它主要把 user-facing tool/protocol 工作放在 SMG，让 engine 看到 token-level request。

3. **vLLM V1 和 TensorRT-LLM 都有独立 CPU 控制面设计。**
   vLLM 是多进程 API/EngineCore/GPU worker；TensorRT-LLM 是 Executor loop + CUDA Graph + Overlap Scheduler。

4. **需要避免的写法：**
   不应写“TokenSpeed 有 C++ Scheduler，所以 agent CPU 高负载一定更低”。更准确的写法是“TokenSpeed 把 scheduler/KV 决策放进 C++，但 agent CPU 路径还包括 SMG、AsyncLLM、EventLoop、OutputProcessor 与 Python materialization”。

### 2.7 CPU 控制面的待验证计数器

如果后续要回答“到底谁解决了 CPU 高负载”，至少要拆这些计数器；本文目前只做架构对照，不声称这些计数器已经验证：

| 计数器 | TokenSpeed 归因位置 | vLLM 归因位置 | TensorRT-LLM 归因位置 |
|---|---|---|---|
| template/tokenization CPU time | SMG / AsyncLLM | API Server Process | LLM API / serving layer |
| scheduler step CPU time | C++ Scheduler + EventLoop overhead | EngineCore scheduler | Executor/PyExecutor scheduler |
| input tensor materialization time | ModelExecutor / InputBuffers | GPU worker model input preparation | ModelEngine input preparation |
| kernel launch gap | ModelExecutor / CUDA graph wrapper | GPU worker / CUDA graph path | CUDA Graph |
| output postprocess time | Generation OutputProcessor + AsyncLLM OutputProcessor + SMG parser | EngineCore/API Server output path | Sampler/output handling |
| tool loop overhead | SMG MCP / external app | API/server/external app | serving/external app |
| p95 ITL gap attribution | EventLoop overlap 是否覆盖 CPU work | EngineCore/API split + worker loop | Overlap Scheduler |

## 3. 相同问题二：Agent KV 管理如何处理

### 3.1 问题本身

带 tool call 的 agent session 通常不是同一条 request 原地暂停/恢复，而是多轮请求围绕同一段历史上下文反复构造 prompt：

```mermaid
flowchart LR
  R1["Round 1 prompt<br/>system + tools + user"]
  P1["prefill prefix"]
  D1["short decode"]
  T1["tool_call output"]
  EXT["external tool execution"]
  R2["Round 2 prompt<br/>history + tool result"]
  P2["prefill mostly repeated prefix"]
  D2["short decode"]

  R1 --> P1 --> D1 --> T1 --> EXT --> R2 --> P2 --> D2
```

KV 管理要回答的不是“有没有 cache”，而是：

- repeated prefix 如何命中？
- request finish 后哪些 KV 可以成为后续 prefix？
- KV 接近满载时如何 evict / retract / offload / recompute？
- 多 replica 时下一轮请求是否还能落到有 KV locality 的 worker？
- tool call 等待期间是否存在专门 pause/resume 或 DDR offload？

### 3.2 TokenSpeed 的 KV 管理路径

```mermaid
flowchart TB
  subgraph S["C++ Scheduler / CPU logical state"]
    REQ["Request FSM<br/>Submitted / Prefilling / Decoding / Draining / Retracted / Finished"]
    PFX["KVPrefixCache / HybridPrefixCache"]
    OWN["OwnedPages / LocalKVAllocator"]
    RPA["ReqPoolAllocator"]
    PLAN["ExecutionPlan<br/>ForwardOp + CacheOp"]
    REQ --> PFX
    REQ --> OWN
    REQ --> RPA
    REQ --> PLAN
  end

  subgraph PY["Python worker materialization"]
    EL["EventLoop"]
    MEM["MemoryExecutor<br/>loadback/writeback when enabled"]
    MX["ModelExecutor"]
    RTP["req_to_page"]
    IB["InputBuffers / out_cache_loc"]
    EL --> MEM
    EL --> MX --> RTP
    MX --> IB
  end

  subgraph GPU["GPU physical state"]
    POOL["token_to_kv_pool"]
    ATT["attention backend"]
    POOL --> ATT
  end

  PLAN -->|"CacheOp"| MEM
  PLAN -->|"ForwardOp"| MX
  RTP --> ATT
  IB --> ATT
  ATT -->|"output tokens"| EL
  EL -->|"Extend / Finish / Reserve"| REQ
```

**实现事实：**

- C++ Scheduler 的 request state 携带资源所有权，不只是状态枚举。
- `OwnedPages` 是 move-only RAII page owner，减少 early-free / double-free / ownership 漏转移风险。
- prefix match、device/host page depth、loadback diff、decode reserve、finish insert、writeback/retract 都在 scheduler/FSM 语义里。
- `ExecutionPlan` 同时包含 `ForwardOp` 和 `CacheOp`；Python EventLoop 把二者提交给 ModelExecutor / MemoryExecutor。
- GPU 实际 KV tensor 在 `token_to_kv_pool`；C++ Scheduler 管 logical page ownership，`ModelExecutor` 通过 `req_to_page` 和 `InputBuffers` materialize。
- tool call 本身不会让 engine 进入一个专门状态；tool result 后通常是下一轮新 request，靠 prefix cache / routing / prompt stability 复用。
- DeepSeek V4 当前 path 禁用 hierarchical KVStore，因此 host/L2/L3 loadback/writeback 不应算作 V4 已经可用的 agent KV 优化。

### 3.3 vLLM V1 的 KV 管理路径

```mermaid
flowchart TB
  subgraph EC["EngineCore / Scheduler"]
    S["Unified Scheduler"]
    BUD["token budget"]
    PRE["preemption / recompute policy"]
    S --> BUD
    S --> PRE
  end

  subgraph KV["KVCacheManager"]
    GCB["get_computed_blocks()"]
    AS["allocate_slots()"]
    CB["cache_blocks()"]
    FR["free()"]
    GCB --> AS --> CB
    AS --> FR
  end

  subgraph BP["BlockPool"]
    BLK["KVCacheBlock<br/>block_id / hash / ref_cnt"]
    FQ["free queue"]
    HM["hash -> block id"]
    BLK --> FQ
    BLK --> HM
  end

  subgraph EXT["Optional KV extensions"]
    HYB["Hybrid KV groups / KVCacheCoordinator"]
    CONN["KV connector / disaggregated prefill"]
  end

  S --> KV
  KV --> BP
  KV --> EXT
```

**官方文档事实：**

- vLLM V1 prefix cache 在 KV cache manager 中实现。
- `KVCacheBlock` 包含 immutable block id、block hash、reference count 和 free queue 指针。
- KV cache manager 初始化时预分配 block pool，并维护 free queue、hash key 到 block id、request id 到 block id 的映射。
- scheduler 会调用 `get_computed_blocks()` 查找已计算 prefix，再调用 `allocate_slots()` 分配新 blocks / touch cached blocks / evict LRU blocks。
- request finished 时，ref count 为 0 的 blocks 会被释放进 free queue；cached block 在 free queue head 被重新分配时会被 eviction。
- hybrid KV cache manager 用 `KVCacheCoordinator` 协调 per-group managers。
- disaggregated prefill 通过 connector / buffer / pipe 等抽象转移 KV cache。

### 3.4 TensorRT-LLM 的 KV 管理路径

```mermaid
flowchart TB
  subgraph EX["Executor / PyExecutor loop"]
    S["Scheduler"]
    CS["CapacityScheduler<br/>fit / pause active requests"]
    S --> CS
  end

  subgraph KV["KVCacheManager / KVCacheConfig"]
    PAGE["paged KV cache"]
    REUSE["KV cache reuse"]
    OFF["host offload<br/>host_cache_size"]
    SALT["cache salt / UUID identifiers"]
    EVT["KV cache events"]
    PAGE --> REUSE
    PAGE --> OFF
    PAGE --> SALT
    PAGE --> EVT
  end

  subgraph GPU["TensorRT Engine / GPU"]
    TRT["ModelEngine"]
    MEM["GPU KV memory"]
    TRT --> MEM
  end

  CS --> KV
  KV --> GPU
  EVT -->|"external cache-aware routing / management"| EX
```

**官方文档事实：**

- TensorRT-LLM KV cache system 通过 `KVCacheConfig` 控制 dtype、GPU memory fraction、host offload 等行为。
- `host_cache_size` 可以启用 host memory offload：block 在从 GPU eviction 前可 offload 到 host，复用时再拷回 GPU。
- KV cache salting 控制哪些请求可以共享 cached KV state，用于隔离不同用户/tenant。
- multimodal UUID 可作为 KV cache event 中的稳定标识。
- scheduler 文档中 `CapacityScheduler` 会根据 KV cache capacity 等资源决定哪些 active requests 适合本步执行，并输出 fitting_requests / paused_requests。
- Executor/architecture 文档中 KVCacheManager 与 Scheduler 在 PyExecutor loop 内协同。

### 3.5 KV 管理实现对照图

```mermaid
flowchart TB
  Q["同一问题：多轮 agent prefix + KV pressure + finish/abort/reuse"]

  subgraph TS["TokenSpeed"]
    TS1["Request FSM owns resource state"]
    TS2["KVPrefixCache / HybridPrefixCache"]
    TS3["OwnedPages / LocalKVAllocator"]
    TS4["ExecutionPlan: ForwardOp + CacheOp"]
    TS5["ModelExecutor materializes req_to_page"]
    TS1 --> TS2 --> TS3 --> TS4 --> TS5
  end

  subgraph VL["vLLM V1"]
    VL1["Scheduler token budget"]
    VL2["KVCacheManager"]
    VL3["BlockPool / KVCacheBlock"]
    VL4["free queue + ref count + block hash"]
    VL5["Hybrid KV / connector"]
    VL1 --> VL2 --> VL3 --> VL4
    VL2 --> VL5
  end

  subgraph TRT["TensorRT-LLM"]
    TR1["Executor Scheduler"]
    TR2["CapacityScheduler"]
    TR3["KVCacheManager / KVCacheConfig"]
    TR4["paged KV reuse"]
    TR5["host offload / events / salting"]
    TR1 --> TR2 --> TR3 --> TR4 --> TR5
  end

  Q --> TS
  Q --> VL
  Q --> TRT
```

| 对照点 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| KV 基本单位 | page / paged cache group / prefix tree node | block / KVCacheBlock | page/block in KVCacheManager |
| request 与 KV 的绑定 | Request FSM 状态对象携带 allocator/node ref/page ownership | scheduler request 与 KVCacheManager/BlockPool 协同 | Scheduler/CapacityScheduler 与 KVCacheManager 协同 |
| prefix reuse | scheduler prefill 时 prefix match；finish 后可 insert prefix tree | `get_computed_blocks()` + block hash + prefix cache | KV cache reuse，cache salting/UUID 控制共享边界 |
| finish 后处理 | Finish/Draining/WritingBack/Finished 状态推进；是否 writeback 取决于 cache path | `free()` ref count 为 0 blocks；cached blocks 留在 free queue 等待 LRU eviction | cache reuse/offload/event 机制由 KVCacheConfig/manager 控制 |
| KV pressure | retract/loadback/writeback 语义存在；DSV4 hierarchical path 当前受限 | preemption/recompute、LRU eviction、connector/disagg path | CapacityScheduler、paused_requests、host offload |
| tool call 关系 | engine 不感知 tool call object；下一轮靠 prefix/routing | API/tool path 与 engine KV cache 分层 | serving/API 与 Executor request 分层 |
| host/DDR offload | 通用 KVStore/MemoryExecutor 路径存在；DSV4 当前不能计入 | V1 guide 显示 GPU <> CPU KV cache swapping removed；另有 connector/disagg mechanisms | host_cache_size 控制 host offload |
| 外部可观测/路由 | SMG routing 可影响 worker locality；engine KV event 另需具体闭环 | KV events/connector docs存在；本文未深挖路由实现 | KV cache events 可用于外部 cache-aware routing/management |

### 3.6 同一条 agent 请求的 KV 生命周期对照

下面把 Round 1 / tool wait / Round 2 放到同一张生命周期图里。这里重点看 engine/executor 层如何处理 KV，而不是 tool parser 如何识别 JSON。

```mermaid
flowchart LR
  A["Round 1 prefill<br/>system + tools + user"]
  B["Round 1 decode<br/>tool_call tokens"]
  C["Round 1 finish"]
  D["Tool wait<br/>external execution"]
  E["Round 2 prefill<br/>history + tool result"]
  F["Round 2 decode<br/>final answer"]

  A --> B --> C --> D --> E --> F

  subgraph KQ["KV questions at each boundary"]
    Q1["A: prefix match?"]
    Q2["C: generated tokens enter reusable cache?"]
    Q3["D: KV kept, evicted, offloaded, or just cached?"]
    Q4["E: second request hits same worker/cache?"]
    Q5["pressure: preempt/retract/offload/recompute?"]
  end
```

| 生命周期阶段 | TokenSpeed | vLLM V1 | TensorRT-LLM |
|---|---|---|---|
| Round 1 prefill prefix lookup | C++ Scheduler 做 prefix match；必要时考虑 device/host depth 和 loadback diff，但 DSV4 host path 受限 | KVCacheManager `get_computed_blocks()` 查 prefix cached blocks | KVCacheManager 做 KV reuse；共享边界受 cache salt / request identity 等配置影响 |
| Round 1 prefill 新 KV 写入 | `ForwardOp` 经 ModelExecutor 写 `token_to_kv_pool`，logical pages 由 Scheduler ownership 管 | scheduler 分配 slots，GPU worker 写 KV blocks | TensorRT engine 写 paged KV cache |
| Round 1 decode tail / reserve | Scheduler 管 decode reserve；output feedback 用 `UpdateReserveNumTokens` 等事件推进 | scheduler 维护 running request 与 allocated blocks；spec/chunked prefill 等由 token budget 管 | Scheduler/CapacityScheduler 管 active request 资源；paged KV 随 executor step 更新 |
| tool_call finish | Generation OutputProcessor 发 Finish/Extend 事件回 C++ Scheduler；SMG 解析 tool_call | request finished 后 KV manager 释放/缓存 blocks，API 层解析或返回 tool_call | request 完成后 response 返回，KV cache manager 可保留可复用 KV |
| tool wait | 当前未看到 engine 内部专用 `ToolWait` 状态；通常是 request finish 后外部 tool 执行 | engine request 完成；外部 app/server 执行 tool | executor request 完成；外部 app/server 执行 tool |
| wait 期间 device KV | prefix tree/cache policy 决定保留/释放/写回；DSV4 不应声称 DDR pause/resume | cached blocks 可留在 block pool/free queue，LRU eviction 时释放；V1 guide 不支持 GPU<>CPU swap | 可通过 host offload 延长 KV 存活，具体取决于 KVCacheConfig 和 eviction/offload policy |
| Round 2 request routing | SMG routing 可影响是否回到相同 worker/cache locality | routing/serving 层决定是否回到有 cache 的 worker；本文未深挖 | KV event API 可服务外部 cache-aware routing；具体看部署 |
| Round 2 prefix hit | 如果 prompt rendering 稳定且 worker/cache 命中，Scheduler prefix match 可减少 prefill | APC hash/block match 可减少 prefill | KV reuse 可减少重复 prefix compute |
| KV pressure | retract/writeback/loadback 机制存在；DSV4 hierarchical path 当前不能计入 | LRU eviction、preemption/recompute、connector/disagg path | CapacityScheduler paused_requests、host offload、KV events |

### 3.7 三框架 KV 状态迁移图

这一张图用同一个抽象状态描述三者的 KV 生命周期差异。它不是源码状态名逐字映射，而是帮助报告读者看到“KV 从哪里来、到哪里去”。

```mermaid
stateDiagram-v2
  [*] --> PromptTokens
  PromptTokens --> PrefixLookup
  PrefixLookup --> NewKVWrite: miss or partial hit
  PrefixLookup --> ReuseKV: hit
  NewKVWrite --> ActiveDecode
  ReuseKV --> ActiveDecode
  ActiveDecode --> FinishedRequest
  FinishedRequest --> ReusableCache: cache retained
  FinishedRequest --> Freed: no retention / evicted
  ReusableCache --> Round2PrefixHit
  ReusableCache --> Evicted: pressure
  ReusableCache --> Offloaded: host/offload path
  Offloaded --> Reloaded: later reuse
  Reloaded --> Round2PrefixHit
  Evicted --> Recompute
  Freed --> Recompute
```

| 抽象状态 | TokenSpeed 对应 | vLLM V1 对应 | TensorRT-LLM 对应 |
|---|---|---|---|
| `PrefixLookup` | `KVPrefixCache.Match()` / hybrid prefix match | `get_computed_blocks()` | KV cache reuse lookup |
| `NewKVWrite` | `ForwardOp` + `req_to_page` + `out_cache_loc` 写 `token_to_kv_pool` | `allocate_slots()` 后 GPU worker 写 blocks | TensorRT engine 写 paged KV |
| `ActiveDecode` | `Decoding` FSM + decode reserve | running request + allocated blocks | active request in scheduler |
| `FinishedRequest` | `FinishEvent` -> Draining/Finished | request finished -> `free()`/cache blocks | response complete in executor |
| `ReusableCache` | prefix tree node resource / cached pages | block hash + cached blocks in block pool/free queue | reusable KV cache pages |
| `Offloaded` | 通用 KVStore/MemoryExecutor path；DSV4 当前受限 | V1 guide 中 GPU<>CPU swap removed；external connector/disagg 另论 | host offload via `host_cache_size` |
| `Round2PrefixHit` | Round 2 same prompt prefix + worker locality + prefix match | APC block hash hit | KV reuse hit / cache-aware routing |
| `Recompute` | prefix miss 后 prefill recompute | prefix miss or preemption recompute | miss/eviction 后 recompute |

这张图能澄清几个容易混淆的点：

1. **tool wait 不是天然等于 KV pause。**
   对三个框架来说，外部 tool 执行期间 engine request 通常已经完成；是否保留 KV 取决于 cache manager、routing、capacity、offload policy，而不是 tool_call 这个语义本身。

2. **Round 2 是否受益取决于两个条件。**
   prompt prefix 要稳定，且请求要命中含有可复用 KV 的 worker/cache。如果 gateway 每轮重新路由到不同 worker，engine prefix cache 能力会被削弱。

3. **TokenSpeed 的 `ExecutionPlan` 差异是实现组织差异。**
   它把 `ForwardOp + CacheOp` 放进同一轮 scheduler plan；这说明 cache movement 与 forward scheduling 有明确接口，但不自动说明真实 workload 下收益更高。

4. **TensorRT-LLM 的 host offload 是文档明确能力。**
   但这属于 NVIDIA runtime 下的 KVCacheConfig 能力；不能直接推断到 Ascend。

5. **vLLM V1 不是弱 KV baseline。**
   APC、BlockPool、Hybrid KV、connector/disaggregated prefill 都是已有架构元素。比较时不能把 vLLM 简化为“只有普通 paged cache”。

### 3.8 KV 生命周期的待验证问题

本文当前能回答“各自如何实现”，但还不能回答“哪个更有效”。要回答后者，至少需要这些事实：

| 问题 | TokenSpeed 需要证明 | vLLM 需要对照 | TensorRT-LLM 需要对照 |
|---|---|---|---|
| Round 2 prefix hit | SMG prompt 稳定 + worker locality + Scheduler prefix match 是否稳定 | APC hash hit / cached token ratio | KV reuse hit / cache event/routing |
| finish 后 KV retention | generated tokens 是否进入 prefix tree，保留多久 | blocks free 后是否仍可被 prefix cache 命中 | reusable KV pages retention / eviction |
| tool wait cache 存活 | tool wait 长度增加时 KV 是否被 eviction/retract | LRU/free queue 行为 | host offload / priority eviction 行为 |
| KV pressure | retract/writeback 是否比 recompute 更好；DSV4 path 是否可用 | preemption/recompute cost | CapacityScheduler pause/offload cost |
| routing locality | SMG routing 是否让同 conversation 回到同 worker | serving/router 是否 cache-aware | KV events 是否接入 cache-aware router |
| p95 TTFT/ITL | prefix reuse + EventLoop overlap 是否移动尾延迟 | EngineCore/APC 是否移动同一指标 | Overlap Scheduler/offload 是否移动同一指标 |

### 3.9 证据索引：阶段到材料

| 对照阶段 | TokenSpeed 本地证据 | vLLM 公开证据 | TensorRT-LLM 公开证据 |
|---|---|---|---|
| 整体进程/组件 | `01-runtime-architecture.md`、`python/tokenspeed/runtime/engine/*`、`tokenspeed-scheduler/csrc/*` | Architecture Overview 的 process architecture | Architecture Overview 的 LLM / PyExecutor / backend loop |
| tool/parser 边界 | `07-smg-gateway-runtime.md`、`02-request-lifecycle.md` 中 T0/T0.5 | API server / serving 层资料；本文未把 tool parser 归入 EngineCore | LLM API / serving 层资料；Executor API 不执行外部 tool |
| CPU 控制面 | `event_loop.py`、`request_handler.py`、`generation_output_processor.py`、`async_llm.py` | Architecture Overview、V1 Guide、V1 blog 对 CPU overhead / EngineCore 的描述 | Architecture Overview、Executor API、Overlap Scheduler 描述 |
| prefix cache | `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/*`、`03-agent-kv-management.md` | Automatic Prefix Caching | KV Cache System / KV cache reuse docs |
| KV block/page ownership | `owned_pages.*`、`kv_allocator.*`、`forward_states.h` | KVCacheBlock / BlockPool / KVCacheManager 文档 | KVCacheManager / KVCacheConfig 文档 |
| KV offload / pressure | `MemoryExecutor`、`KVStore`、`03` 中 DSV4 限制 | V1 Guide、disaggregated prefill / connector 文档 | KV Cache System 中 host offload，Scheduler 中 CapacityScheduler |
| Round 2 routing locality | `07-smg-gateway-runtime.md` 的 routing policy 分析 | 本文未深挖 vLLM router | KV cache events / cache-aware routing 相关官方博客和文档 |

## 4. 当前文档可支持的客观观察

以下不是竞争力结论，只是从架构图得到的差异描述：

1. **TokenSpeed 把 agent-facing gateway 和 engine 分开。**
   SMG 处理 OpenAI/tool/parser/routing；engine 处理 token-level scheduling/KV/model forward。这个分层会影响如何分析 CPU overhead：不能把 parser 成本算到 C++ Scheduler，也不能把 engine KV reuse 归因给 parser。

2. **三者都有 CPU control-plane 设计。**
   TokenSpeed 是 SMG + EventLoop + C++ Scheduler；vLLM V1 是 API Server + EngineCore + GPU Worker 多进程；TensorRT-LLM 是 Executor/PyExecutor + CUDA Graph + Overlap Scheduler。仅凭“有 C++ Scheduler”不能推出 TokenSpeed 特别。

3. **三者都有成熟 KV 管理元素。**
   TokenSpeed 有 request FSM + page ownership；vLLM 有 KVCacheManager / BlockPool / APC / hybrid KV；TensorRT-LLM 有 KV reuse / offload / events / CapacityScheduler。不能把“有 prefix cache”写成 TokenSpeed 特点。

4. **TokenSpeed 的可讨论差异在语义耦合方式，而不是功能名。**
   具体是 `Request FSM -> KV ownership -> ExecutionPlan -> ModelExecutor materialization -> output feedback -> scheduler.advance` 这一条链路。但这只是实现差异，不自动等于性能优势。

5. **DeepSeek V4 / tool call 场景必须降权 host offload 叙事。**
   代码分析没有看到 tool-call-specific DDR KV offload；DeepSeek V4 hierarchical KVStore 当前受限。因此报告中只能说“通用 runtime 有 cache movement 机制”，不能说“V4 agent tool wait 已支持 DDR KV pause/resume”。

## 5. 面向 PPT 的图页建议

如果把这部分拆成 PPT，不建议先讲“TokenSpeed 哪里更强”。建议按如下图页组织：

1. **三框架整体架构图**
   一页三列：TokenSpeed / vLLM V1 / TensorRT-LLM，标出 gateway/API、scheduler loop、KV manager、GPU execution 的边界。

2. **同一条 agent 请求的系统边界**
   用 `messages + tools -> tool_call -> tool result -> Round 2` 串起 gateway/API、engine/executor、GPU forward、parser/tool runtime。

3. **同一问题：agent CPU 高负载**
   用一张公共问题图说明 CPU overhead 来源，再用阶段表说明三者在 template/tokenize、admission、scheduler step、GPU launch、output/parser、tool wait 的实现位置。

4. **同一问题：agent KV 管理**
   用一张 multi-turn tool loop 图说明 repeated prefix，再用生命周期表对比 Round 1 prefill、tool_call finish、tool wait、Round 2 prefix hit、KV pressure。

5. **事实对照表，而非结论表**
   表头用“组件边界 / 状态所有权 / offload 支持 / tool call 边界 / 未验证项”，不要用“强/弱/结论”。

6. **最后只列开放问题**
   例如：真实 CPU time breakdown、prefix hit under tool loop、worker locality、DSV4 hierarchical KVStore 是否可落地、Ascend 上 MoE dispatch-combine 等价实现。

## 6. 外部资料索引

- vLLM architecture overview: https://docs.vllm.ai/en/stable/design/arch_overview/
- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide/
- vLLM prefix caching: https://docs.vllm.ai/en/v0.11.1/design/prefix_caching/
- vLLM hybrid KV cache manager: https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- vLLM disaggregated prefill: https://docs.vllm.ai/en/v0.12.0/features/disagg_prefill/
- TensorRT-LLM architecture: https://nvidia.github.io/TensorRT-LLM/architecture/overview.html
- TensorRT-LLM Executor API: https://nvidia.github.io/TensorRT-LLM/advanced/executor.html
- TensorRT-LLM scheduler: https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html
- TensorRT-LLM KV cache system: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
