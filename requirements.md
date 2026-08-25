# BRISK requirements

## Supported operating environment

| Requirement | Qualified environment |
| --- | --- |
| Host operating system | Linux supported by AMD Vitis and AMD XRT |
| Hardware build tools | AMD Vitis and Vivado 2025.2 with `v++` available |
| Runtime used for qualification | AMD XRT 2.18.179 with Python `pyxrt` |
| FPGA platform | `xilinx_u50_gen3x16_xdma_5_202210_1` |
| FPGA device | Xilinx Alveo U50 selected by `DEVICE` |
| GPU used by the live comparison | NVIDIA GeForce RTX 5090 visible to PyTorch CUDA |
| Python packages | `jupyter`, `ipywidgets`, `numpy`, `torch`, `transformers`, and `pyxrt`; official Meta checkpoints also require `fairscale`, `tiktoken`, and the sibling `llama` source package |

## Local Llama model

The demo uses local model material and does not download weights.

| Model format | Local requirement |
| --- | --- |
| Hugging Face Llama3-8B | A local directory with `config.json`, compatible model weights, and tokenizer material |
| Official Meta Llama3-8B | A local directory with `params.json`, `consolidated.00.pth`, and `tokenizer.model`, plus a sibling official `llama` source package |
| Attention geometry | 32 query heads, 8 K/V heads, and 128 dimensions per head |
| Public prompt envelope | Batch 1 with an active sequence from 1 through 8 tokens |

## Release artifact identity

| Release field | Qualified value |
| --- | --- |
| GitHub release | `vic9112/BRISK`, tag `v1.0.0` |
| Asset | `BRISK-U50-v1.0.0.hw.xclbin` |
| Size | 50,176,274 bytes |
| SHA-256 | `d928f26881fe59588ff7481f0945fcd857a276dffbe6b0f5dbfd12690de44ded` |
| Platform VBNV | `xilinx_u50_gen3x16_xdma_5_202210_1` |
| xclbin UUID | `3a241b4c-830e-1d8c-b01b-a0883977d31f` |
| Static interface UUID | `16e2362f-82d2-feab-3552-9da27134b76d` |
| Packaged kernels | 1 |
| DATA clock | 220 MHz |
| HBM clock | 400 MHz |

## Ordered kernel ABI

The release exposes the `llama3_fast_attention` kernel with this ordered ABI. Buffer groups and HBM mapping are defined by `source/link.cfg`.

| Index | Argument | Kind |
| ---: | --- | --- |
| 0 | `q0` | BF16 query buffer |
| 1 | `q1` | BF16 query buffer |
| 2 | `q2` | BF16 query buffer |
| 3 | `q3` | BF16 query buffer |
| 4 | `active_kv` | BF16 paired active key/value buffer |
| 5 | `cache_kv0` | Persistent key/value cache buffer |
| 6 | `cache_kv1` | Persistent key/value cache buffer |
| 7 | `cache_kv2` | Persistent key/value cache buffer |
| 8 | `cache_kv3` | Persistent key/value cache buffer |
| 9 | `out0` | BF16 output buffer |
| 10 | `out1` | BF16 output buffer |
| 11 | `out2` | BF16 output buffer |
| 12 | `out3` | BF16 output buffer |
| 13 | `batch_size` | Scalar integer |
| 14 | `seqlen` | Scalar integer |
| 15 | `start_pos` | Scalar integer |
| 16 | `cache_stride` | Scalar integer |
| 17 | `cache_valid_len` | Scalar integer |
| 18 | `mask_mode` | Scalar integer |
| 19 | `flags` | Scalar integer |
| 20 | `cache_identity` | Scalar integer |
| 21 | `status_out` | Scalar-status output buffer |

## Semantic boundary

| Boundary | Contract |
| --- | --- |
| Input | Post-RoPE BF16 query, key, and value tensors for dense causal grouped-query attention |
| Arithmetic | FP32 scoring, true exponential, online softmax recurrence, and value accumulation |
| Cache ownership | Kernel-owned append and versioned authoritative K/V residency, with exact raw-HBM fallback for a nonresident identity |
| Output | BF16 pre-output-projection attention context plus a status code |
| Excluded substitutions | No approximation, pruning, sparsity, quantization, or reduced internal precision |
| Demo comparison | Same prompt-derived first-layer boundary on RTX 5090 and U50; host invocation and device timing scopes remain separate |

## Routed implementation

| Routed completion metric | Value |
| --- | ---: |
| Kernel period | 4.545 ns |
| Kernel frequency | 220 MHz |
| Whole-design setup WNS / TNS | 0.000 ns / 0.000 ns |
| Whole-design hold WHS / THS | 0.000 ns / 0.000 ns |
| Failed / partially routed / unrouted nets | 0 / 0 / 0 |
| Bus-skew constraints passed | 225 / 225 |
| Minimum bus-skew slack | 1.832 ns |

| Whole routed device resource | Used | Available | Utilization |
| --- | ---: | ---: | ---: |
| CLB LUTs | 360,600 | 870,720 | 41.41% |
| CLB registers | 552,120 | 1,741,440 | 31.70% |
| Block RAM tiles | 800 | 1,344 | 59.52% |
| URAM | 164 | 640 | 25.63% |
| DSP48E2 | 1,805 | 5,952 | 30.33% |
| CLBs | 85,685 | 108,840 | 78.73% |

| Routed SLR resource | SLR0 used | SLR0 utilization | SLR1 used | SLR1 utilization |
| --- | ---: | ---: | ---: | ---: |
| CLBs | 47,962 | 87.27% | 37,723 | 69.86% |
| CLB LUTs | 201,857 | 45.91% | 158,743 | 36.75% |
| CLB registers | 333,684 | 37.95% | 218,436 | 25.28% |
| Block RAM tiles | 441 | 65.63% | 359 | 53.42% |
| URAM | 84 | 26.25% | 80 | 25.00% |
| DSPs | 921 | 31.98% | 884 | 28.78% |

## Fresh board qualification

| Correctness or ownership gate | Qualified shape | Calls or identities | Maximum absolute error | Threshold | Status / numerical / nonfinite violations |
| --- | --- | ---: | ---: | ---: | ---: |
| Base prefill and decode continuation | Batch 1, sequence 8 at position 0; sequence 1 at positions 8 through 18 | 12 calls | Within BF16 gate | 0.0625 | 0 / 0 / 0 |
| 32-layer incremental continuation | Batch 1, 8 tokens, chunk 1, 32 layers | 256 calls | 0.015625 | 0.0625 | 0 / 0 / 0 |
| Public prompt-derived first-layer Attention | Batch 1, original 32-layer Llama3-8B, 4 prompt tokens | 1 prompt | 0.00048828125 | 0.125 | 0 / 0 / 0 |
| Device-active captured-tensor profile | Batch 2, sequence 8 at position 0; sequence 1 at positions 8 through 18 | 360 samples | 0.000977 | Absolute 0.0625 and relative 0.08 | 0 / 0 / 0 |

| Cache identity gate | Identities | Historical HBM reads | HBM writes | Cold-fallback HBM reads | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident prefill and poison-backed decode | 32 | 0 bytes | 0 bytes | 0 bytes | 0 |
| Different-identity fallback after poison | 1 | Not resident | 0 bytes | 32,768 bytes in 512 transactions | 0 |

| Device-active profile method | Value |
| --- | ---: |
| Warm-up chains | 5 |
| Measured launches per workload | 30 |
| Workloads | 12 |
| Total measured launches | 360 |

Device-active latency is derived only from XRT ERT command-state `RUNNING` and `COMPLETED` timestamps. Host launch-to-completion is reported separately.

| Mode | Batch | Sequence | Start position | Samples | Device-active mean | Device-active minimum | Device-active maximum | Launch-to-completion mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 2 | 8 | 0 | 30 | 39.231 us | 33.913 us | 41.460 us | 107.376 us |
| Decode | 2 | 1 | 8 | 30 | 31.043 us | 29.981 us | 31.825 us | 55.278 us |
| Decode | 2 | 1 | 9 | 30 | 31.272 us | 30.065 us | 33.759 us | 51.659 us |
| Decode | 2 | 1 | 10 | 30 | 31.402 us | 30.201 us | 32.885 us | 61.574 us |
| Decode | 2 | 1 | 11 | 30 | 31.226 us | 29.665 us | 33.066 us | 80.145 us |
| Decode | 2 | 1 | 12 | 30 | 31.418 us | 30.699 us | 33.133 us | 130.832 us |
| Decode | 2 | 1 | 13 | 30 | 31.368 us | 30.299 us | 33.085 us | 126.221 us |
| Decode | 2 | 1 | 14 | 30 | 31.284 us | 29.750 us | 33.166 us | 111.221 us |
| Decode | 2 | 1 | 15 | 30 | 31.256 us | 30.305 us | 32.166 us | 49.610 us |
| Decode | 2 | 1 | 16 | 30 | 31.221 us | 29.181 us | 33.739 us | 77.707 us |
| Decode | 2 | 1 | 17 | 30 | 31.093 us | 30.419 us | 31.949 us | 46.869 us |
| Decode | 2 | 1 | 18 | 30 | 31.393 us | 30.040 us | 38.206 us | 83.247 us |

## Build and demo storage

| Location or behavior | Requirement |
| --- | --- |
| `build/` | Generated Vitis products, qualification output, logs, downloaded release artifact, and reports; ignored by Git |
| Local override | Optional untracked `local.mk` or command-line Make variables |
| Default demo artifact order | Existing local `build/llama3_fast_attention.hw.xclbin`, then the checked GitHub release asset |
| Explicit demo artifact | `XCLBIN=/path/to/file.hw.xclbin`; a missing explicit path fails without downloading another file |
| Release integrity gate | SHA-256 must match before the xclbin is passed to the notebook |
