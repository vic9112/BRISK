# BRISK

## Contribution summary

BRISK is a reproducible, memory-first U50 implementation of exact dense causal grouped-query attention. The public source keeps BF16 tensor boundaries, FP32 internal arithmetic, and the true exponential used by online softmax. The accompanying notebook compares the same local prompt-shaped attention boundary on a pure RTX path and a hybrid CPU, GPU, and U50 path.

## Quick start

```bash
make
make hls
MODEL_DIR=... make demo
make clean
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `source/` | Humanized HLS source, U50 connectivity, and physical Tcl hooks |
| `demo/` | Local live-attention notebook and its runtime |
| `Makefile` | One-command staged hardware build and local-demo entry points |
| `requirements.md` | Environment, ABI, boundary, and artifact requirements |

## Award statement

| Statement | Status |
| --- | --- |
| FPGenius was prepared for the FPT'26 Design Competition Track B. | Submission provenance; this prerelease makes no award claim. |

## Qualified results

Prior-source qualification only; a fresh BRISK rebuild and board run are pending. These historical values do not identify a v1.0.0 release artifact.

| Prior-source qualification | Value | Qualification status |
| --- | --- | --- |
| Routed kernel request | 239 MHz / 4.184 ns | Historical routed result; pending fresh rebuild |
| Routed WNS / WHS | 0.012 ns / 0.004 ns | Historical routed result; pending fresh rebuild |
| Maximum per-SLR LUT / FF | 42.52% / 35.51% | Historical routed result; pending fresh rebuild |
| Maximum per-SLR BRAM / DSP / URAM | 33.41% / 31.63% / 1.25% | Historical routed result; pending fresh rebuild |
