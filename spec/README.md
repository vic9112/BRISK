# BRISK HLS specification

Start with [the attention kernel](llama3_fast_attention.md), then the functions
that own its record service and arithmetic.

| Specification | Responsibility |
| --- | --- |
| [llama3_fast_attention](llama3_fast_attention.md) | Host interface, causal attention, task coordination and output stores. |
| [serve_kv](serve_kv.md) | Single-owner K/V residency, install-and-forward, nonallocating history fallback and displacement. |
| [score_queries](score_queries.md) | FP32 grouped query/key scores. |
| [normalize_scores](normalize_scores.md) | FP32 online softmax factors and final sums. |
| [accumulate_values](accumulate_values.md) | FP32 weighted-value accumulation and BF16 output. |

[Payload and callable declarations](brisk_types.hpp) and
[stream aliases](brisk_stream_types.hpp) define the public interfaces; they are
declarations, not replacement implementations. Each specification includes
Architecture, Dataflow and State views, with editable HTML sources beside the
SVG figures.

The paper's datapath and single-owner contribution remain the basis of this
specification. Host packing, caller lifecycle, finite-input support, numerical
check conditions and buffering details are explicitly identified implementation
contracts where the paper is silent. Logical work and supply targets are not
measured throughput, physical module counts or proofs of FIFO adequacy. New
correctness and performance claims require fresh implementation evidence.

## Paper reference

Kuan-Hsi Chen, Yan-Jhih Wang, Jui-Chi Yu, and Hsi-Pin Ma. “BRISK: Bypassing Repeated I/O via Single-Owner K/V Residency for Llama 3 Attention on FPGA.”

Architecture and numerical behavior: §II, pp. 1–2. Evaluation boundary and
workloads: §III, p. 2. No manuscript PDF is redistributed in this directory.
