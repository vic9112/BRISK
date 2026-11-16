# BRISK HLS source guide

This guide starts with the selected kernel path. Long, uncalled alternatives
are kept in separate headers so a reviewer need not follow them through the
active implementation. Short compatibility helpers remain explicitly marked.

The cleanup changes indentation, comments and where unselected definitions are
kept. It does not change active expressions, types, function boundaries, loop
traversal, pragma contents, interface mappings or buffer capacities. Readability
alone is not evidence of correctness, hardware identity or performance.

## Follow the active path

Start at `llama3_fast_attention` in [attention_kernel.cpp](attention_kernel.cpp).
After scalar validation, it forms bounded row tiles and calls `run_tile`.

| Responsibility | Active functions | Source |
| --- | --- | --- |
| Validate the call and cover host rows with local tiles | `validate_attention_call`, `llama3_fast_attention`, `make_tile_plan` | [attention_kernel.cpp](attention_kernel.cpp), [attention.hpp](attention.hpp) |
| Distribute incoming paired K/V words | `load_active_tile`, `write_active_word` | [attention_kernel.cpp](attention_kernel.cpp) |
| Connect a lane's input, resident owner and scoring tasks | `run_lane_front` | [attention_kernel.cpp](attention_kernel.cpp) |
| Load that lane's queries | `load_q` | [qk_datapath.hpp](qk_datapath.hpp) |
| Forward incoming lines into the lane owner | `update_cache_tile` | [attention_kernel.cpp](attention_kernel.cpp) |
| Retain lane-local K/V and select resident or backing data | `load_kv` | [kv_stream.hpp](kv_stream.hpp) |
| Compute canonical FP32 scores with temporal row reuse | `qk_lane`, `canonical_qk_products`, `reduce_eight` | [kv_stream.hpp](kv_stream.hpp), [qk_datapath.hpp](qk_datapath.hpp) |
| Service the lanes' online normalization | `online_sm`, `online_sm_step` | [kv_stream.hpp](kv_stream.hpp) |
| Update weighted values | `sv_lane` | [kv_stream.hpp](kv_stream.hpp) |
| Connect value accumulation, final normalization and output | `run_lane_back`, `normalize`, `store_output` | [attention_kernel.cpp](attention_kernel.cpp), [online_softmax.hpp](online_softmax.hpp) |

Each `load_kv` lane specialization has its own persistent payload and tag
storage. `update_cache_tile` forwards incoming lines; it does not create a
second authoritative resident store. The existing resident-path qualification,
tag handling, backing policy and continuation logic are preserved here.

QK and SV use flattened action loops with query rows interleaved inside the
dimension-chunk traversal. A selected K or V chunk stays live across its
eligible rows. Query and weighted-value state retain each row's separate
partial results. Keep this traversal when reviewing readability changes:
rewriting it as independent load/compute/output phases changes scheduling even
when the mathematical attention formula is unchanged.

The online normalization produces a scale whose previous or current factor is
unity. `sv_lane` selects the nontrivial factor and its matching operand, retaining
the single-multiply update. Final normalization computes a reciprocal and
multiplies the packed output elements by it. Preserve this operation order;
elementwise division is not the same implementation.

## Reading the loops and diagnostics

The source uses consistent block indentation, including the resident owner's
general path and the QK action loop. Each stage's entry comment describes its
role; comments at the temporal loops identify held data and output conditions.
The flattened loops are intentional, not a request to rewrite the algorithm as
separate load/compute/store phases.

At the boundaries where unselected definitions were moved, `#line` restores the
original logical source coordinates. HLS uses those coordinates in generated
operation and loop names. The directives are source-location metadata, not
runtime controls or extra hardware. Active bodies keep their original line
breaks, so those names can be compared directly rather than renamed or ignored.
A diagnostic's logical line may therefore differ from the editor's physical
line; use its function name and expression to locate it. Do not add further
line-reflow changes without checking the generated hardware again.

## Geometry and element units

| Quantity | Submitted value or derivation |
| --- | --- |
| Query heads / K/V heads / lanes | `32 / 8 / 4` |
| K/V heads per lane | `2 = 8 / 4` |
| Query representatives per K/V head | `4 = 32 / 8` |
| Query heads per lane | `8 = 2 * 4` |
| Head dimensions / local query-row bound | `128 / 16` |
| Query/output word | `256` bits, containing `16` BF16 elements |
| Query/output words per head | `8 = 128 / 16` |
| Paired K/V word | `256` bits, containing `8` BF16 K elements and `8` BF16 V elements |
| Paired K/V words per head | `16 = 128 / 8` |
| Resident/backing line | `512` bits, containing `2` paired K/V words |
| Wide lines per K/V head / lane record | `8 / 16` |
| SIMD dimensions / chunks per head | `8 / 16` |
| Resident positions | `1280 = 32 layers * 40 positions per layer` |
| Per-batch resident window | `40` for one batch; `20` for each of two batches |
| Logical resident payload across lanes | `4 * 1280 * 16 * 64 = 5242880` bytes |

The interface `depth` attributes count elements of the declared pointer type,
not bytes. The active top-level pragmas are the source of these counts.

| Pointer family | Depth | Element unit and derivation |
| --- | --- | --- |
| Q/output per lane | `1048576` | `256`-bit words: `2 * 8192 * 8 local query heads * 8 words` |
| Incoming active K/V | `2097152` | `256`-bit paired words: `2 * 8192 * 8 K/V heads * 16 words` |
| Backing K/V per lane | `262144` | `512`-bit lines: `2 * 8192 * 2 local K/V heads * 8 lines` |
| Status | `1` | One status element |

The scalar bounds, indexing, element types and pointer extents must agree.
An interface depth is not a measured burst length or memory bandwidth.

## Prefill frame staging

The `prefill_physical_frame` guard selects the existing contiguous-frame
load/store paths. It does not substitute a different attention operation.
These paths stage contiguous memory before supplying the temporal computation;
their loop structure and access order remain unchanged.

| Literal or frame size | Meaning in the selected path |
| --- | --- |
| `2` batches and `8` new rows | Shape selected by the prefill frame guard |
| `16` flattened tokens | `2 batches * 8 new rows` |
| Query frame of `512` packed words | Per batch: `8 rows * 8 local query heads * 8 words` |
| Query frame load of `256` adjacent pairs | `512 packed words / 2` |
| Incoming frame of `64` wide lines per token | `8 K/V heads * 16 paired words / 2` |
| Incoming token stride of `128` paired words | `8 K/V heads * 16 paired words` |
| Output frame of `64` packed words per lane/token | `8 local query heads * 8 words` |
| Output frame store of `32` adjacent pairs | `64 packed words / 2` |

## Buffering and mapping choices

These are retained implementation choices, not additional model dimensions.
The value FIFO decouples K/V delivery from the score-to-scale startup path.
Other FIFOs absorb the selected producer/consumer skew. Their presence does
not establish minimum depth or universal stall safety.

| Source stream | Submitted capacity | Packet unit |
| --- | --- | --- |
| Representative query streams | `8` each | Query pair |
| Even and odd incoming streams | `16` each | Paired K/V word |
| Forwarded incoming records | `16` | Wide resident/backing line |
| Keys | `4` | K dimension chunk |
| Values | `128` | V dimension chunk |
| Scores | `8` | Representative-score packet |
| Scales | `64` | Normalization-factor packet |
| Denominators | `32` | Completed-query sum packet |
| Accumulators | `2` | Weighted-value chunk packet |
| Normalized output | `64` | Packed query/output word |

The declared `ready` stream is unused by the active producer and consumer;
do not count it as a realized channel without inspecting generated hardware.

| Other selected setting | Interpretation |
| --- | --- |
| Required/provisioned value-flight chunks | The inherited assertion uses `112 / 128`; the former is a pipeline-flight threshold, not a head-geometry derivation |
| Resident payload partition factor | `5`, organizing `1280` positions into banks of `256` positions and `4096` wide-line addresses |
| Q/output outstanding-request limit | `8` read and `8` write requests in the top interface pragmas |
| Active-input outstanding-request limit | `8` read requests in the top interface pragma |
| Configured maximum Q/output and active burst length | `256` beats; not an assertion that every access attains that burst |

Matching synthesis, scheduling, burst and dataflow reports are required before
changing these choices. Preserve the submitted pipeline targets and dependency
directives while establishing an unchanged baseline. This guide makes no new
II, primitive-utilization, deadlock-freedom or board-performance claim.

## Retained definitions outside the active path

Long unselected bodies are separated from the selected path. They remain
included for source compatibility; including a header does not select another
datapath. These are source-level reachability observations from
`llama3_fast_attention`, not a replacement for the synthesized call graph.

| Retained alternate source | Contents outside the selected path |
| --- | --- |
| [qk_alternatives.hpp](qk_alternatives.hpp) | Task-ordered query loading, hybrid operands and `stream_qk` |
| [kv_alternatives.hpp](kv_alternatives.hpp) | `load_resident_kv` and `sv_update` |
| [softmax_alternatives.hpp](softmax_alternatives.hpp) | `stream_softmax_interleaved` |
| [attention_alternatives.hpp](attention_alternatives.hpp) | Anonymous-namespace cache streaming and `accumulate_values`, with their local dependencies |
| [shared_product.hpp](shared_product.hpp) | Hybrid product-bank definitions; not the selected canonical QK arithmetic |

Do not start a review in those headers or treat their constants as active
optimization controls. They are retained reference/compatibility code, not
additional modes exposed by the selected kernel.

| Definition or helper family | Current role |
| --- | --- |
| `FATTN_QO_INTERFACE_DEPTH` | Active in the Q/output interface pragmas |
| `FATTN_ACTIVE_DEPTH`, `FATTN_CACHE_DEPTH` | Retained split-plane counts; the active paired-data interface pragmas use their own depths |
| `FATTN_COMPACT_B2S8_CACHE_STRIDE` | Retained definition, not read by the current top or its frame guard |
| `FATTN_BOUND_WORKLOAD` and corresponding bound arguments | Binding is disabled; runtime caller arguments are selected |
| Dual-head aliases, model-width and active-plane counts | Retained metadata definitions with no active use |
| Plane-layout, dual-port/elastic-credit and halfbeat helpers | Retained alternate organization or static checks, not the current active transport |
| Hybrid/FIP product-bank helpers | Unselected alternate arithmetic; the active QK path uses canonical products |
| `FATTN_LIPR_TAG_BITS` | Retained logical tag-width count; not used to declare the tag fields |
| `FATTN_STATUS_CACHE_BUSY` | Retained status declaration, not returned by the current top |
| `stream_qk`, `stream_softmax_interleaved`, `load_resident_kv`, `sv_update` | Uncalled alternate function bodies |
| `stream_cache_tile` and anonymous-namespace `accumulate_values` | Uncalled alternatives in the top source; active SV is `sv_lane` |

Do not reinterpret the retained definitions as controls for a new datapath.
Definitions are retained, not silently deleted or renamed. The stored
reciprocal-square-root literal and BF16 conversion masks likewise remain
unchanged rather than being replaced with newly evaluated arithmetic.

This cleanup also leaves existing boundary validation, cache/tag behavior and
known verification limitations unchanged. It is not a functional bug-fix pass,
and does not extend support to controls or cache transitions not established by
the submitted implementation's evidence.
