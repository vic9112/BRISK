# HLS source guide

Read [the public contract](../spec/README.md), then [attention_kernel.cpp](attention_kernel.cpp). Its responsibility headers implement the declared task boundaries. Public types and callables come from [brisk_types.hpp](../spec/brisk_types.hpp) and [brisk_stream_types.hpp](../spec/brisk_stream_types.hpp), not duplicate implementation definitions.

| Responsibility | Source | Contract |
| --- | --- | --- |
| Validate controls, tile host buffers, distribute inputs and drain outputs | [attention_kernel.cpp](attention_kernel.cpp) | [Kernel](../spec/llama3_fast_attention.md) |
| Own resident K/V, select history and preserve live victims | [serve_kv.hpp](serve_kv.hpp) | [K/V service](../spec/serve_kv.md) |
| Reuse queries/keys and compute causal grouped scores | [score_queries.hpp](score_queries.hpp) | [QK](../spec/score_queries.md) |
| Maintain independent maxima/sums and emit matched factors | [normalize_scores.hpp](normalize_scores.hpp) | [Normalization](../spec/normalize_scores.md) |
| Reuse values, update FP32 weighted sums, divide and pack | [accumulate_values.hpp](accumulate_values.hpp) | [SV](../spec/accumulate_values.md) |
| Shared geometry, derived bounds and arithmetic helpers | [attention.hpp](attention.hpp) | Public type, precision and geometry rules. |

The kernel includes the responsibility headers; they share `attention.hpp`. Blocking HLS streams preserve payload order and association under backpressure. There is no additional software-managed availability or credit protocol.

## Geometry and ownership

| Quantity | Value or derivation |
| --- | --- |
| Model geometry | `32` query heads, `8` K/V heads and head dimension `128`. |
| Static lane owners | `4` independent compile-time lane specializations, each owning `2` K/V heads and their `8` query heads. |
| Query sharing | `4` query representatives per K/V head. |
| Layer / batch / host-sequence / tile bounds | `32` layers; batch `1`–`2`; host sequence up to `8192`; local tile up to `16` rows. |
| Resident positions | `40` per layer, divided equally across active batches; batch-two therefore receives `20` per batch. |
| Q/O word | `256` bits, containing `16` BF16 elements. |
| Paired K/V pack | `256` bits, containing `8` K and `8` V elements. |
| Cache line | `512` bits, containing adjacent paired packs. |
| Chunks / lines per head | `128/8 = 16` chunks; `16/2 = 8` lines. |
| Lines per lane record | `2*8 = 16`, covering both local K/V heads under the same ownership tag. |
| Maximum logical Q/O extent per port | `1,048,576` packed words. |
| Maximum logical active-K/V extent | `2,097,152` paired packs. |
| Maximum logical backing extent per lane | `262,144` cache lines. |
| Q/active read-transport settings | `8` outstanding reads and maximum `256`-beat bursts; intended latency hiding, not measured bandwidth. |

Each static lane specialization owns independent resident payload, tag and layer-context arrays. The selected layer's ring is partitioned by batch. History misses read backing without allocation or tag changes. Incoming installation preserves a live victim completely before reusing its slot; a new valid tag follows full installation. An accepted fresh context retires only the selected layer. Refer to the contract for cold bootstrap, unchanged-identity reuse and append-only continuation.

## Buffering and arithmetic

The chosen FIFO depths are integration settings, not proved minimum depths. Counts apply to each lane stream, including each representative-specific query stream.

| Channel | Depth in tokens | Purpose |
| --- | ---: | --- |
| Incoming cache lines | 16 | Stage a local K/V record. |
| Query pairs, per representative | 8 | Decouple query fetching from tile reuse. |
| Key chunks | 4 | Absorb short source/scorer skew. |
| Value chunks | 128 | Decouple source service from value updates. |
| Score packets | 8 | Absorb QK/normalizer skew. |
| Rescaling packets | 64 | Decouple factors from the matched SV recurrence. |
| Final denominators | 32 | Hold completed tile-row/local-head sums. |
| Output words | 64 | Decouple packed output from external stores. |

Transport and K/V storage preserve raw BF16. QK, standard-exponential normalization, weighted-value updates and final division use FP32. Scores, probabilities and intermediate factors are not converted to BF16. Only final output conversion uses BF16 round-to-nearest, ties-to-even.

## Configuration ownership and checks

| File | Responsibility |
| --- | --- |
| [hls_config.cfg](hls_config.cfg) | Production HLS configuration, including all non-status M_AXI interface settings and declared memory depths. Do not duplicate those directives in C++. |
| [link.cfg](link.cfg) | Platform connectivity and HBM mapping. |
| `tcl/` | Physical implementation hooks used by the Makefile. |
| [verification](../verification/README.md) | Independent numerical, memory, validation and complete RTL case-group checks. |

Use `make hls`, `make hls-cosim` and `make vivado` from the repository root. A source fence or passing C simulation does not by itself establish completed external writes, finite-channel safety or routed timing. Review generated request/response behavior and completed RTL transactions; board correctness and performance require execution of the built artifact.

| Default generated location | Contents |
| --- | --- |
| `build/hls_component/` | Production HLS component, including its synthesis reports and generated hardware. |
| `build/hls_adapter/`, `build/hls_identity/` | Bounded co-simulation components; generated hardware must match production. |
| `build/reports/cosim/<suite>/` | Completed suite log, copied native reports and dataflow information. |
| `build/reports/link/`, `build/logs/link/` | Standard implementation reports and logs. |
| `build/report/hw/link/imp/impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt` | Dedicated final routed timing summary written by the implementation hook and used as clock evidence for notebook execution and qualification. |

These locations follow `BUILD_DIR` when it is changed. Platform or clock changes require a fresh build directory or a clean rebuild; Make does not automatically invalidate products for those option changes.

Clock topology metadata can describe an HBM request rather than its effective selection. Only the [version-bound SYSTEM/HBM mapping](../requirements.md#version-bound-hbm-clock-handling) is supported; it is not generic rounding. Effective clocks must meet routed STA, and physical DATA, KERNEL and SYSTEM counters are mandatory before kernel execution. The documented mapping alone is not a measured-clock or board-success result.
