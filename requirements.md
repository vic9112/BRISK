# BRISK requirements

## Environment

| Requirement | Supported baseline or obligation |
| --- | --- |
| Host | Little-endian Linux supported by AMD Vitis and XRT, with access to the selected U50 device. |
| Build tools | AMD Vitis and Vivado 2025.2; `v++` must be available. |
| FPGA runtime | AMD XRT 2.18.179 with compatible Python `pyxrt` and matching loaded `xclmgmt`/`xocl` drivers. |
| FPGA platform | `xilinx_u50_gen3x16_xdma_5_202210_1`. |
| GPU comparison | NVIDIA GeForce RTX 5090 visible to PyTorch CUDA. |
| Python runtime | An environment containing `numpy`, CUDA-enabled `torch`, `transformers`, `ipywidgets` and `pyxrt`. |
| Headless notebook execution | Also requires `nbformat`, `nbclient`, `IPython`, `ipykernel` and a working `python3` Jupyter kernel. It does not require JupyterLab. |
| Official Meta checkpoint support | Also requires `fairscale`, `tiktoken` and the compatible official `llama` source package. |
| Notebook execution | `make demo` uses the headless notebook runner; JupyterLab is optional for interactive viewing. |
| Local storage | Writable space for HLS/co-simulation products, routed implementation, reports and local model material. Build products remain ignored by Git. |

Activate the installed tool environments before running Make. Alternatively, pass setup-script paths through `VITIS_SETUP` and `XRT_SETUP`. These variables are optional when the shell is already configured; they are not paths to bundled tools.

| Make input | Meaning or default |
| --- | --- |
| `MODEL_DIR` | Existing local model directory; required for the demo. |
| `XCLBIN` | Explicit existing artifact, or the default `build/llama3_fast_attention.hw.xclbin` when not supplied. |
| `DEVICE` | XRT device index; default `0`. |
| `PLATFORM` | Installed U50 platform named above. |
| `HLS_FREQ_MHZ` | HLS scheduling target; default `220` MHz, not an achieved-frequency claim. |
| `KERNEL_FREQ_MHZ` | Link/implementation clock target and expected qualification clock; default `220` MHz. |
| `ROUTED_CLOCK_REPORT` | Matching routed timing summary for the selected artifact, required by notebook execution and device qualification. The default dedicated report path is listed in the source guide; supply the matching report for an external artifact whose report is elsewhere. |
| `PYTHON` | Python interpreter used for the demo and helper scripts. |
| `JOBS` | Build parallelism supported by the local machine. |
| `BUILD_DIR` | Generated build/report directory; default `build/`. |

No model-weight or release-artifact download is part of these commands. A missing explicitly selected artifact is an error, not a fallback request.

Build products are incremental and do not track command-line platform or clock changes. Use a fresh `BUILD_DIR`, or clean the existing build directory before changing `PLATFORM`, `HLS_FREQ_MHZ` or `KERNEL_FREQ_MHZ`. The supplied routed report and expected clock must belong to the artifact being executed or qualified.

The notebook runner accepts `--routed-clock-report`, otherwise uses `ROUTED_CLOCK_REPORT` when supplied. Its remaining default is the dedicated routed timing report under the selected xclbin's parent directory. The standard Make build supplies this evidence without an additional demo argument.

| Clock semantic | Required role |
| --- | --- |
| `DATA` | `kernel00` clock. |
| `KERNEL` | `kernel01` clock. |
| `SYSTEM` | Single system/HBM clock. |

Before notebook startup or device programming, the runner checks the artifact's declared clock topology against its matching routed timing contract. After image loading, it samples physical clocks before the actual callback and again afterward. The effective clocks must satisfy the bound routed contract; requested targets or metadata values alone do not establish the device's operating clocks.

### Version-bound HBM clock handling

The supported runtime has a narrowly verified SYSTEM/HBM request mapping. It is not a general clock-rounding rule and does not change the routed timing contract.

| Scope or clock quantity | Supported rule or requirement |
| --- | --- |
| Applicable runtime | XRT `2.18.179` with matching loaded `xclmgmt` and `xocl` drivers. |
| Applicable clock | SYSTEM/HBM only; no inferred rounding for DATA or KERNEL. |
| Routed effective HBM requirement | `400` MHz. |
| Covered HBM metadata-request interval | `[400,405)` MHz, including a `402` MHz request. |
| Version-bound effective selection | Requests in that interval select the `400` MHz HBM setting on the supported runtime. |
| Physical validation | All `3` clock counters are mandatory before kernel execution; effective clocks must match the routed contract. |

A raw metadata request can therefore differ from the selected effective HBM setting. The mapping does not establish a physical measurement: actual DATA, KERNEL and SYSTEM counters must still be available and checked before launching attention. No wider tolerance, other request bucket or other runtime version is implied.

## Local model material

| Model format | Required local material |
| --- | --- |
| Hugging Face Llama3-8B | A directory containing `config.json`, compatible local weights and tokenizer material. Loading uses local files only. |
| Official Meta Llama3-8B | A directory containing `params.json`, `consolidated.00.pth` and `tokenizer.model`; the compatible `llama` Python package is its sibling. |
| Official package contents | The sibling `llama` package supplies `__init__.py`, `generation.py`, `model.py` and `tokenizer.py`. |
| Device-qualification input | `demo/qualify_u50.py` currently uses official Meta checkpoint material and its sibling `llama` source package. Notebook support for Hugging Face does not imply qualification-script support for that format. |
| Attention geometry | Query heads: `32`; K/V heads: `8`; head dimension: `128`. |
| Notebook prompt envelope | Batch `1`; active sequence from `1` through `8` tokens. Longer prompts are limited to the supported active token count by the demo runtime. |

The notebook derives attention inputs from the local model. It is an attention-boundary demonstration, not an end-to-end text-generation or full-model throughput measurement.

## Kernel boundary and ABI

The [kernel specification](spec/llama3_fast_attention.md) is the detailed contract. [Public declarations](spec/brisk_types.hpp) preserve the ordered callable below; [connectivity](source/link.cfg) supplies the memory-bank mapping.

| Index | Argument | Kind |
| ---: | --- | --- |
| 0 | `q0` | Packed BF16 query input. |
| 1 | `q1` | Packed BF16 query input. |
| 2 | `q2` | Packed BF16 query input. |
| 3 | `q3` | Packed BF16 query input. |
| 4 | `active_kv` | Packed paired incoming K/V input. |
| 5 | `cache_kv0` | Persistent backing-memory region. |
| 6 | `cache_kv1` | Persistent backing-memory region. |
| 7 | `cache_kv2` | Persistent backing-memory region. |
| 8 | `cache_kv3` | Persistent backing-memory region. |
| 9 | `out0` | Packed BF16 output. |
| 10 | `out1` | Packed BF16 output. |
| 11 | `out2` | Packed BF16 output. |
| 12 | `out3` | Packed BF16 output. |
| 13 | `batch_size` | Signed integer control. |
| 14 | `seqlen` | Signed integer control. |
| 15 | `start_pos` | Signed integer control. |
| 16 | `cache_stride` | Signed integer control. |
| 17 | `cache_valid_len` | Signed integer control. |
| 18 | `mask_mode` | Signed integer control. |
| 19 | `flags` | Signed integer control. |
| 20 | `cache_identity` | Signed identity encoding. |
| 21 | `status_out` | Status output buffer. |

| Boundary | Contract |
| --- | --- |
| Inputs | Post-RoPE BF16 Q/K/V for dense causal grouped-query attention. |
| Arithmetic | FP32 QK products/reductions, standard exponential, online normalization, weighted-value updates and final division. |
| Output | BF16 attention context before output projection, using round-to-nearest, ties-to-even. |
| K/V authority | Lane-local resident owner; exact-tag resident selection, nonallocating history fallback and completed live-victim writeback before slot reuse. |
| Fresh sequence | An accepted `start_pos=0` retires the selected layer's old logical sequence and clears its resident tags before installation, including unchanged identity reuse. Invalid calls do not retire state. |
| Continuation | Serialized, append-only calls with stable live identity, batch geometry, stride, backing addresses and earlier logical records. |
| Exclusions | No record dropping, sparse/approximate attention, approximate exponential, BF16 score/probability rounding or reduced internal precision. |
| Caller duties | Valid pointer extents, alignment and disjointness; supported finite inference domain; distinct backing records for concurrently live layers. |

| Geometry or control | Supported value |
| --- | --- |
| Layers / lanes | `32` layers; `4` lanes. |
| Query/K/V heads per lane | `8` query heads and `2` K/V heads. |
| Batch / host sequence / local tile | Batch `1`–`2`; host sequence up to `8192`; local tile up to `16` new rows. |
| Residency | `40` positions per layer, shared equally across active batches. |
| Identity | Signed `32`-bit ABI; low `5` bits select the layer and high `27` bits encode the epoch; negative encodings are legal. |
| Paper-conforming flags | `0`. |

## Numerical and timing interpretation

| Comparison | Required interpretation |
| --- | --- |
| General output acceptance | Actual and reference values are finite, with `abs(O-O_ref) <= max(0.0625,0.08*max(abs(O_ref),1e-6))`. The reference is independent FP32 dense causal attention from BF16 inputs, rounded to BF16 at output. |
| Exact analytic cases | Require their independently calculated BF16 encodings in addition to applicable general checks. |
| Notebook ratio | GPU host-observed attention invocation through CUDA completion divided by U50 submission through XRT wait. |
| CUDA-event diagnostic | Report separately; never mix with the notebook host-envelope ratio. |
| Device qualification | Use the matched Accelerator Monitor (AM)/CUDA-graph procedure in `demo/qualify_u50.py`, separately from notebook timing. Its GPU baseline is unfused FP32 dense operators, not optimized SDPA or FlashAttention. |
| Excluded timing work | Projection, RoPE, output projection and host tensor transfers are outside the attention-only device-active boundary. |

These are requirements and measurement definitions, not current timing, resource-utilization or board-success results. Build and co-simulation reports cannot substitute for actual notebook execution or device qualification on the selected artifact.
