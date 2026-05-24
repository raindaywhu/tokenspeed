# TokenSpeed Competitiveness Validation Framework

## 1. Purpose

This document defines how to evaluate whether TokenSpeed has real advantages over vLLM and TensorRT-LLM for agent workloads.

The goal is not to assume that TokenSpeed is better. The goal is to separate:

- verified advantages
- potential advantages
- architectural differences
- parity areas
- disadvantages
- unknowns that require benchmark or profiling evidence

A feature should not be described as an advantage only because TokenSpeed has a different architecture. It should be described as an advantage only when the difference creates measurable benefits under the same workload, model, hardware, and serving constraints.

## 2. What Counts as an Advantage

A TokenSpeed capability should be classified as a verified advantage only if all three conditions are satisfied:

1. Architecture difference

   TokenSpeed has a runtime mechanism that is meaningfully different from vLLM and TensorRT-LLM.

2. Observable benefit

   The mechanism improves a measurable serving property, such as:

   - lower CPU control-plane overhead
   - smaller forward-to-forward gap
   - lower TTFT
   - lower TPOT
   - better Round 2 TTFT in tool-call workflows
   - higher prefix cache hit rate
   - fewer KV evictions or recomputations
   - better GPU utilization under agent traffic
   - better tail latency under mixed prefill/decode/tool workloads

3. Reproducible evidence

   The benefit is reproduced under a controlled benchmark with:

   - same model
   - same hardware
   - same quantization
   - same request distribution
   - same tool schema size
   - same concurrency level
   - comparable tokenizer and chat template behavior
   - comparable routing and cache locality policy

If only condition 1 is satisfied, the result should be called a potential advantage.

If conditions 1 and 2 are plausible but not measured, the result should be called a hypothesis.

If the benefit depends on code paths that are not implemented or not enabled for the target model, the result should be called unverified.

## 3. Result Labels

Use the following labels consistently.

| Label | Meaning |
|---|---|
| Verified advantage | Supported by source code, profiling, and reproducible benchmark results |
| Potential advantage | Supported by architecture analysis, but missing benchmark evidence |
| Parity | Similar capability exists in vLLM or TensorRT-LLM |
| Disadvantage | TokenSpeed is weaker, less mature, or less complete |
| Unknown | Insufficient source or benchmark evidence |
| Not applicable | The comparison dimension does not apply to the selected runtime or model path |

## 4. Decision Matrix

| Dimension | TokenSpeed | vLLM | TensorRT-LLM | Current Judgment |
|---|---|---|---|---|
| OpenAI-compatible protocol handling | Usually handled by SMG gateway or serving adapter | Mature OpenAI-compatible serving path | Usually handled by Triton, custom server, or upper-layer adapter | Not a TokenSpeed engine advantage |
| Anthropic / Claude-style protocol handling | Gateway or adapter layer responsibility | Can be implemented via serving adapter; not EngineCore responsibility | Usually upper-layer responsibility | Protocol support is not an engine-level advantage |
| Tool-call parser | Gateway / parser layer | API serving / ToolParser layer | Usually upper-layer adapter or server integration | Parity or integration difference, not engine advantage |
| Tool-call lifecycle | Tool loop is likely handled outside the engine | Tool parser and tool loop handled outside EngineCore | Usually handled outside runtime backend | Engine-level advantage not proven |
| Same-request tool wait and resume | Not proven | Not default behavior | Not default behavior | Unknown / do not claim |
| Tool-call-specific DDR KV offload | Not proven | V1 does not rely on GPU-CPU KV swapping by default | Depends on KV cache manager and deployment | Unknown / do not claim |
| Multi-round prefix reuse | Potentially possible through prefix cache, routing, and KV lifecycle | Mature prefix caching and KV manager capabilities | Supports KV reuse/offload depending on runtime configuration | Needs benchmark |
| Scheduler control plane | C++ Scheduler and request FSM may reduce overhead | EngineCore scheduler is mature and widely used | C++ runtime with inflight batching and strong GPU execution stack | TokenSpeed potential advantage, not verified |
| CPU fixed overhead in agent workloads | SMG + Python + C++ scheduler mixed path | API server + EngineCore + workers | Triton/server + C++ backend path | Needs profiling |
| Forward-to-forward gap | Potentially reducible if scheduler and output path are efficient | Needs measurement | Usually strong when graph/runtime path is optimized | Needs profiling |
| KV ownership model | TokenSpeed appears to emphasize logical KV ownership and page lifecycle | vLLM has mature KV block manager and prefix cache | TensorRT-LLM has mature KV cache mechanisms | Potential advantage only if measured |
| DeepSeek / MLA specialization | Important TokenSpeed target path | Supported depending on model implementation and kernels | Strong NVIDIA kernel/runtime ecosystem | Unknown without model-specific benchmark |
| MoE and communication maturity | Needs validation across TP/EP/DP/CP combinations | Mature distributed serving ecosystem | Strong NCCL/NVIDIA stack | TokenSpeed likely less mature |
| Production readiness | Preview / early-stage | Mature production usage | Mature production usage | TokenSpeed disadvantage |
| Observability and tooling | Needs explicit counters and profiler support | Mature ecosystem and metrics | Mature deployment/profiling stack | TokenSpeed disadvantage unless improved |
| Ascend portability | Strategically possible if CUDA/NCCL assumptions are isolated | Requires backend support and adaptation | NVIDIA-centric | TokenSpeed potential strategic value, high engineering risk |
| Kernel maturity | Depends on model path and custom kernels | Broad backend support | Very strong NVIDIA kernel stack | TokenSpeed not automatically advantaged |
| Agent workload fit | Potentially strong if scheduler/KV/runtime integration is validated | Strong baseline; cannot be treated as weak | Strong backend, but upper-layer agent loop may be external | Needs controlled benchmark |

## 5. Core Interpretation

TokenSpeed's possible advantage is not generic tool-call support.

Tool-call parsing, protocol conversion, and MCP/tool execution loops are usually gateway or serving-layer responsibilities. vLLM also places tool-call parsing in the OpenAI-compatible serving path rather than in EngineCore or GPU workers.

Therefore, the key question is not:

"Does TokenSpeed support tool calls?"

The key question is:

"Does TokenSpeed's scheduler, request FSM, KV lifecycle, and execution plan reduce the cost of multi-round agent serving compared with vLLM and TensorRT-LLM?"

This requires measurement.

## 6. Competitiveness Claim Ledger

### Claim 1: TokenSpeed C++ Scheduler may reduce agent CPU control-plane overhead

Status: Potential advantage, not verified.

Current evidence:

- TokenSpeed has an embedded C++ Scheduler and request FSM.
- The scheduler appears to own request lifecycle and logical KV page state.
- The design may reduce Python-side scheduling overhead if the critical path is actually moved into C++.

Missing evidence:

- per-step scheduler latency
- forward-to-forward CPU gap
- SMG gateway overhead
- Python materialization overhead
- output processing overhead
- parser and structured output overhead
- comparison against vLLM EngineCore and TensorRT-LLM inflight batching

Required validation:

- same model
- same hardware
- same batch/concurrency
- same tool schema
- same prompt distribution
- CPU flamegraph
- scheduler timeline
- GPU idle gap measurement

Risk of overclaiming:

Do not claim CPU advantage only because TokenSpeed has a C++ Scheduler. Agent workloads may still be dominated by gateway, tokenizer, parser, output processing, or Python orchestration overhead.

### Claim 2: TokenSpeed may have stronger multi-round agent KV lifecycle control

Status: Potential advantage, not verified.

Current evidence:

- TokenSpeed design discusses request FSM, page allocation, retract/loadback/writeback concepts, and KV ownership.
- This could be useful for multi-round agent workloads if KV retention and prefix reuse are coordinated with routing and scheduler decisions.

Missing evidence:

- whether tool-call completion retains KV
- whether Round 2 can reuse Round 1 prefix KV reliably
- whether routing preserves KV locality
- whether KV is evicted, reloaded, or recomputed under pressure
- whether target model paths support hierarchical KVStore
- whether DeepSeek V4 path supports the relevant KV lifecycle features

Required validation:

- Round 1 tool_call followed by mock tool wait and Round 2 tool_result
- measure Round 2 TTFT
- measure prefix cache hit rate
- measure KV retained / evicted / recomputed / reloaded
- measure worker locality hit rate
- compare with vLLM and TensorRT-LLM under the same request distribution

Risk of overclaiming:

Do not describe tool-call-specific DDR KV offload and same-request resume as implemented unless the code path and runtime trace prove it.

### Claim 3: TokenSpeed may be better positioned for agent-specific scheduling

Status: Hypothesis.

Current evidence:

- TokenSpeed has a request FSM and execution-plan-oriented design.
- Agent workloads have different characteristics from normal chat: shorter bursts, more round trips, larger shared tool prefixes, and higher CPU fixed overhead.

Missing evidence:

- whether scheduler policies distinguish agent rounds from normal chat
- whether tool schema or structured output constraints affect scheduling decisions
- whether Round 2 requests receive special locality-aware scheduling
- whether request FSM improves fairness, tail latency, or cache retention

Required validation:

- mixed workload benchmark with normal chat and agent traffic
- P50/P90/P99 latency
- queueing delay
- scheduler decision trace
- per-request cache lifecycle trace

Risk of overclaiming:

Do not claim agent-aware scheduling unless the scheduler actually observes and uses agent-relevant state.

### Claim 4: TokenSpeed may be strategically useful for Ascend adaptation

Status: Strategic hypothesis, high engineering risk.

Current evidence:

- TokenSpeed separates scheduling, model execution, and gateway concerns.
- It is not inherently tied to TensorRT-LLM's NVIDIA-only runtime stack at the same level.

Missing evidence:

- HCCL replacement plan for NCCL assumptions
- CUDA stream/event equivalent behavior
- CUDA graph replacement or fallback strategy
- custom kernel migration plan
- MLA/MoE kernel availability
- memory allocator compatibility
- distributed executor compatibility
- performance evidence on Ascend

Required validation:

- minimal decode path on Ascend
- attention kernel benchmark
- MoE dispatch benchmark
- scheduler-worker communication benchmark
- KV cache allocation and reuse benchmark

Risk of overclaiming:

Architecture portability does not imply kernel portability. Ascend adaptation risk is concentrated in communication, memory management, custom kernels, and graph execution.

### Claim 5: TokenSpeed may be better for DeepSeek-style MLA/MoE workloads

Status: Unknown.

Current evidence:

- TokenSpeed appears to target specific large-model inference paths such as Kimi/DeepSeek-style workloads.
- It may include model-specific execution and kernel integration.

Missing evidence:

- comparable DeepSeek/MLA/MoE benchmark against vLLM and TensorRT-LLM
- exact supported model path
- kernel maturity
- TP/EP/DP/CP combination support
- memory footprint comparison
- decode throughput and latency comparison
- long-context behavior

Required validation:

- same DeepSeek-style model
- same quantization
- same sequence length
- same batch/concurrency
- same GPU count
- prefill and decode separated
- MoE communication counters
- attention/KV memory counters

Risk of overclaiming:

Do not generalize from one optimized model path to all LLM inference workloads.

## 7. Minimal Agent Benchmark Protocol

### 7.1 Goal

The benchmark must evaluate agent-serving behavior, not only generic tokens per second.

The key measurements are:

- CPU control-plane overhead
- forward-to-forward gap
- Round 1 TTFT
- Round 2 TTFT
- tool-call parser overhead
- tool loop overhead
- prefix cache hit rate
- KV retention and eviction
- GPU utilization
- tail latency under concurrency

### 7.2 Workload A: Baseline Chat

Purpose:

Measure normal serving overhead without tools.

Request shape:

- system prompt
- one user message
- no tools
- max_tokens = 256

Metrics:

- TTFT
- TPOT
- output tokens per second
- GPU utilization
- API/server overhead
- scheduler time
- forward-to-forward gap
- CPU utilization

Interpretation:

This establishes the non-agent baseline. Any agent advantage must be compared against this baseline.

### 7.3 Workload B: Tool Schema Without Tool Execution

Purpose:

Measure the cost of rendering and processing large tool schemas when the model does not call a tool.

Request shape:

- system prompt
- user message
- tools list with 8 / 32 / 128 tools
- tool_choice = auto
- prompt instructs model to answer normally without tool use

Metrics:

- chat template rendering time
- tokenizer time
- prompt token count
- prefix cache hit rate
- TTFT increase versus Workload A
- CPU preparation overhead
- memory footprint

Interpretation:

This isolates the cost of agent/tool schema overhead before any tool call happens.

### 7.4 Workload C: Two-Round Tool Call

Purpose:

Measure the real agent loop.

Round 1:

- messages + tools
- model emits tool_call

Gateway/tool loop:

- parse tool_call
- execute mock tool
- append tool_result

Round 2:

- messages + previous assistant tool_call + tool_result
- model emits final answer

Metrics:

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

Interpretation:

This is the most important benchmark for agent workloads. If TokenSpeed has an advantage, it should appear in Round 2 TTFT, CPU gap, prefix reuse, or cache lifecycle behavior.

### 7.5 Workload D: Shared Long Prefix Multi-Session Agent Traffic

Purpose:

Measure prefix cache and routing behavior under concurrent agent workloads.

Request shape:

- long shared system prompt
- shared tool schema
- many concurrent sessions
- repeated Round 1 / Round 2 loops
- varied user suffixes

Metrics:

- prefix cache hit rate
- cache eviction rate
- Round 2 TTFT distribution
- P50/P90/P99 latency
- DP rank skew
- worker locality
- CPU utilization
- GPU utilization
- memory pressure

Interpretation:

This benchmark tests whether TokenSpeed can turn shared agent prefixes into real serving efficiency.

### 7.6 Workload E: Mixed Normal Chat and Agent Traffic

Purpose:

Measure whether agent workloads interfere with normal chat workloads.

Request shape:

- 70% normal chat
- 30% two-round tool-call traffic
- shared system prompt
- mixed output lengths

Metrics:

- normal chat TTFT/TPOT
- agent Round 2 TTFT
- scheduler fairness
- queueing delay
- P99 latency
- cache pressure
- GPU utilization

Interpretation:

A production serving engine must handle mixed traffic. A specialized agent optimization should not severely degrade normal chat.

## 8. Required Instrumentation

The following counters or traces are required before making a strong competitiveness claim:

### Gateway / API Layer

- request validation time
- chat template rendering time
- tokenizer time
- detokenizer time
- tool parser time
- structured output parser time
- streaming response overhead
- protocol conversion overhead

### Engine / Scheduler Layer

- request admission time
- queueing time
- scheduler step time
- cache allocation time
- preemption/retract time
- execution plan build time
- output processing time
- forward-to-forward gap

### KV Cache Layer

- prefix cache hit/miss
- KV blocks allocated
- KV blocks retained after request completion
- KV evictions
- KV recomputations
- KV reload/loadback
- host/device KV movement
- cache locality by worker or DP rank

### GPU Layer

- forward time
- attention time
- MLP/MoE time
- communication time
- sampling time
- GPU utilization
- memory bandwidth
- HBM usage
- kernel launch overhead

### Distributed Layer

- DP routing decision
- TP/PP/EP group configuration
- inter-worker communication time
- all-reduce/all-gather time
- MoE dispatch/combine time
- rank skew

## 9. Safe Conclusion Templates

### 9.1 Conservative Conclusion

Based on current source-level analysis, TokenSpeed shows potential architectural differentiation in its C++ Scheduler, request FSM, and KV lifecycle design. However, there is not yet enough benchmark or profiling evidence to conclude that it is generally superior to vLLM or TensorRT-LLM for agent workloads.

The most appropriate current conclusion is that TokenSpeed deserves deeper evaluation under controlled agent benchmarks.

### 9.2 Strong but Still Safe Conclusion

TokenSpeed's most promising advantage is not generic tool-call support. Its possible advantage lies in tighter integration between request FSM, logical KV ownership, scheduler decisions, and execution-plan materialization.

If profiling confirms lower forward-to-forward CPU gap, better Round 2 prefix reuse, and lower tail latency under multi-round agent workloads, TokenSpeed could have a meaningful advantage for high-concurrency agent serving.

### 9.3 Negative or Cautious Conclusion

Current evidence is insufficient to claim that TokenSpeed is superior to vLLM or TensorRT-LLM. TokenSpeed may have promising architecture, but vLLM and TensorRT-LLM have more mature production ecosystems, stronger observability, and more proven deployment paths.

Without profiling and benchmark evidence, TokenSpeed should be described as a high-potential preview system rather than a proven superior runtime.

### 9.4 What Not to Say

Do not say:

- TokenSpeed already has stronger tool-call support than vLLM.
- TokenSpeed engine directly understands OpenAI or Anthropic tool calls.
- Tool wait automatically triggers DDR KV offload and same-request resume.
- DeepSeek V4 hierarchical KVStore is already a proven advantage.
- C++ Scheduler automatically means lower CPU overhead.
- TokenSpeed is better than TensorRT-LLM without kernel-level benchmark.
- vLLM is weak in agent serving without a fair benchmark.
- SMG gateway capabilities are the same as TokenSpeed engine capabilities.

### 9.5 Recommended Positioning

TokenSpeed should be positioned as a high-potential inference runtime whose possible advantage lies in scheduler, KV lifecycle, and runtime integration for agent workloads.

Current evidence supports further evaluation, but not a final superiority claim over vLLM or TensorRT-LLM.

## 10. Recommended Next Steps

1. Add source-level citations for every major TokenSpeed claim.
2. Add a runtime diagram separating SMG gateway, TokenSpeed engine, scheduler, and GPU execution.
3. Add counters for scheduler time, cache operations, parser time, and forward-to-forward gap.
4. Implement Workload C first because it best represents real tool-call agent behavior.
5. Compare against vLLM before comparing against TensorRT-LLM.
6. Only after source and benchmark evidence is available, upgrade potential advantages to verified advantages.
