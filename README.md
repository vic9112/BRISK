# BRISK

BRISK — *Bypassing Repeated I/O via Single-Owner K/V Residency for Llama 3 Attention on FPGA* — is a memory-first Alveo U50 implementation of exact dense causal grouped-query attention. It keeps BF16 tensor boundaries, FP32 internal arithmetic, true exponential, and online softmax while retaining repeated K/V state at one authoritative hardware owner.

## Award

| Competition | Track | Result |
| --- | --- | --- |
| FPT'26 Design Competition | Track B | First Prize |

## Quick start

```bash
make
make hls
MODEL_DIR=/path/to/Meta-Llama-3-8B make demo
make clean
```

`make` builds the HLS kernel, U50 implementation, bitstream, and xclbin. `make hls` stops after the XO. `make demo` prefers the local build; when it is absent, it downloads the checked release asset and verifies its SHA-256 before opening the notebook. Set `XCLBIN=/path/to/file.hw.xclbin` to use an explicit local artifact.

## Qualified release

| Artifact field | Qualified value |
| --- | --- |
| Release asset | `BRISK-U50-v1.0.0.hw.xclbin` |
| Size | 50,176,274 bytes |
| SHA-256 | `d928f26881fe59588ff7481f0945fcd857a276dffbe6b0f5dbfd12690de44ded` |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` |
| xclbin UUID | `3a241b4c-830e-1d8c-b01b-a0883977d31f` |
| DATA / HBM clock | 220 MHz / 400 MHz |
| Routed setup WNS / hold WHS | 0.000 ns / 0.000 ns |
| Failed / partially routed / unrouted nets | 0 / 0 / 0 |

## Measured U50 behavior

All rows below were measured on the rebuilt artifact identified above. The latency boundary is the complete post-RoPE, pre-output-projection Attention kernel; projections, RoPE, output projection, and host transfers are excluded from device-active time.

| Board gate | Workload | Measured result |
| --- | --- | ---: |
| Prefill and continuation | Batch 1, prefill length 8, then decode positions 8 through 18 | 12 calls; 0 status failures |
| 32-layer continuation | Batch 1, 8 one-token calls per layer | 256 calls; maximum absolute error 0.015625; threshold 0.0625 |
| Public prompt-derived first-layer Attention | Original 32-layer Llama3-8B, batch 1, 4 prompt tokens | Maximum absolute error 0.00048828125; threshold 0.125; PASS |
| Resident-HBM poison | 32 distinct layer identities | 0 historical read bytes; 0 mismatches |
| Different-identity fallback | One cold identity after poison | 32,768 read bytes; 512 read transactions; 0 mismatches |
| Device-active profile | Batch 2, 12 workloads, 30 measured launches each | 360 samples; 0 numerical violations; 0 nonfinite values; maximum absolute error 0.000977 |

| Workload group | Samples | Mean device-active latency | Mean XRT launch-to-completion latency |
| --- | ---: | ---: | ---: |
| Prefill, batch 2 and sequence 8 | 30 | 39.231 us | 107.376 us |
| Decode, batch 2 and cache positions 8 through 18 | 330 | 31.270 us | 79.487 us |

Device-active latency uses XRT ERT `RUNNING`-to-`COMPLETED` timestamps. Launch-to-completion is the host-observed XRT invocation envelope. The notebook separately compares the same prompt-derived first-layer Attention tensors on a real RTX 5090 and U50 and rejects nonfinite values or an absolute error above its declared threshold.

## Repository layout

| Path | Purpose |
| --- | --- |
| `source/` | Humanized HLS source, U50 connectivity, and physical Tcl hooks |
| `demo/` | Standalone local prompt-derived RTX 5090/U50 Attention notebook |
| `Makefile` | One-command hardware build and checked demo entry point |
| `requirements.md` | Environment, ABI, semantic boundary, and detailed measurements |

Licensed under Apache-2.0.
