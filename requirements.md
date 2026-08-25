# BRISK requirements

## Supported operating environment

| Requirement | Supported environment |
| --- | --- |
| Host operating system | Linux supported by AMD Vitis and AMD XRT |
| Hardware build tools | AMD Vitis and Vivado 2025.2 with `v++` available |
| Runtime stack | AMD XRT with Python `pyxrt` available to the selected interpreter |
| FPGA platform | `xilinx_u50_gen3x16_xdma_5_202210_1`, or a compatible U50 platform supplied through `PLATFORM` |
| FPGA device | Xilinx Alveo U50 selected by `DEVICE` |
| GPU used by the live comparison | NVIDIA RTX 5090 visible to PyTorch CUDA |
| Python packages | `jupyter`, `ipywidgets`, `numpy`, `torch`, `transformers`, and `pyxrt`; official Meta checkpoints also require `fairscale`, `tiktoken`, and the sibling `llama` source package |

## Local Llama model

The demo uses only local model material and does not download weights.

| Model format | Local requirement |
| --- | --- |
| Hugging Face Llama3-8B | A local directory with `config.json`, compatible model weights, and tokenizer material |
| Official Meta Llama3-8B | A local directory with `params.json`, `consolidated.00.pth`, and `tokenizer.model`, plus a sibling official `llama` source package |
| Attention geometry | 32 query heads, 8 KV heads, and 128 dimensions per head |
| Live prompt envelope | Batch 1 with an active sequence from 1 through 8 tokens |

## Ordered kernel ABI

The `llama3_fast_attention` kernel has this exact ordered ABI. Buffer groups and HBM mapping are defined by `source/link.cfg`.

| Index | Argument | Kind |
| ---: | --- | --- |
| 0 | `q0` | BF16 query buffer |
| 1 | `q1` | BF16 query buffer |
| 2 | `q2` | BF16 query buffer |
| 3 | `q3` | BF16 query buffer |
| 4 | `active_kv` | BF16 paired active key/value buffer |
| 5 | `cache_kv0` | persistent key/value cache buffer |
| 6 | `cache_kv1` | persistent key/value cache buffer |
| 7 | `cache_kv2` | persistent key/value cache buffer |
| 8 | `cache_kv3` | persistent key/value cache buffer |
| 9 | `out0` | BF16 output buffer |
| 10 | `out1` | BF16 output buffer |
| 11 | `out2` | BF16 output buffer |
| 12 | `out3` | BF16 output buffer |
| 13 | `batch_size` | scalar integer |
| 14 | `seqlen` | scalar integer |
| 15 | `start_pos` | scalar integer |
| 16 | `cache_stride` | scalar integer |
| 17 | `cache_valid_len` | scalar integer |
| 18 | `mask_mode` | scalar integer |
| 19 | `flags` | scalar integer |
| 20 | `cache_identity` | scalar integer |
| 21 | `status_out` | scalar-status output buffer |

## Semantic boundary

| Boundary | Contract |
| --- | --- |
| Input | Post-RoPE BF16 query, key, and value tensors for dense causal grouped-query attention |
| Arithmetic | FP32 scoring, online softmax recurrence, true exponential, and value accumulation |
| Output | BF16 attention output tensors and a status code |
| Excluded substitutions | No approximation, pruning, sparsity, quantization, or reduced internal precision |
| Demo comparison | Same prompt-shaped first-layer attention boundary on the pure RTX and hybrid U50 paths; the ratio uses comparable host-observed invocation envelopes |

## Generated build storage

| Location | Expectation |
| --- | --- |
| `build/` | Large local generated Vitis and Vivado products, logs, temporary files, and reports; ignored by Git |
| Local configuration | Optional untracked `local.mk` or command-line Make variable overrides |

## Release artifact identity

| Release identity | Current state |
| --- | --- |
| v1.0.0 xclbin download and content hash | Pending Task 6; no release artifact is identified by this prerelease |
