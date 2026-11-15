# BRISK

BRISK — *Bypassing Repeated I/O via Single-Owner K/V Residency for Llama 3 Attention on FPGA* — implements dense causal grouped-query attention on Alveo U50. Tensor boundaries and retained K/V use BF16; scoring, standard-exponential online softmax and value accumulation use FP32. Incoming K/V is installed and forwarded by its authoritative owner; nonresident history uses nonallocating HBM fallback.

Start with the [HLS specification](spec/README.md). The [source guide](source/README.md), [verification guide](verification/README.md) and [environment requirements](requirements.md) explain implementation and execution.

## Build and run

Activate Vitis and XRT in the shell, and provide a local supported Llama checkpoint. Model weights and release binaries are not downloaded.

```bash
make
make hls
make hls-cosim
make vivado
MODEL_DIR=/local/model make demo
```

| Command | Work performed |
| --- | --- |
| `make` | Default full build: HLS/XO generation, the complete RTL case-group union, then routed implementation and bitstream/xclbin generation. |
| `make hls` | C simulation, HLS synthesis and XO generation. |
| `make hls-cosim` | Complete RTL co-simulation case groups, including stateful continuations and validation cases. |
| `make vivado` | Build missing prerequisites, then route and generate the bitstream/xclbin. |
| `MODEL_DIR=/local/model make demo` | Build the default artifact if needed, execute all notebook cells and invoke the actual notebook button callback headlessly. JupyterLab is not required. |
| `MODEL_DIR=/local/model make qualify` | Run the separate device-active AM/CUDA-graph comparison with the supported qualification model input. |

`/local/model` is a placeholder for an existing local checkpoint. To select an existing artifact explicitly:

```bash
MODEL_DIR=/local/model XCLBIN=/path/to/kernel.hw.xclbin \
  ROUTED_CLOCK_REPORT=/path/to/matching-routed-timing.rpt make demo
```

An explicitly supplied `XCLBIN` must exist and have matching routed-clock evidence. A missing explicit path fails; it does not select a downloaded or alternate binary. The default artifact is `build/llama3_fast_attention.hw.xclbin`, whose normal build supplies the report automatically. The runner checks the routed clock contract and artifact clock topology before notebook execution, then samples physical clocks before and after the actual callback.

`VITIS_SETUP` and `XRT_SETUP` may name setup scripts when the shell has not already sourced them. `DEVICE`, `PLATFORM`, `PYTHON`, `JOBS` and `BUILD_DIR` select the corresponding local resources; see [requirements](requirements.md). Generated products and reports remain under the ignored build directory.

Make is incremental. Changing `PLATFORM`, `HLS_FREQ_MHZ` or `KERNEL_FREQ_MHZ` does not invalidate existing products automatically: select a fresh `BUILD_DIR`, or clean the existing build directory before rebuilding. Use the same approach when changing tool installations. Requested clocks are build targets, not measured timing results.

## What the demo measures

The notebook uses the same prompt-derived, post-RoPE attention tensors on the GPU and U50, before output projection. Its displayed ratio compares host-observed invocation envelopes. CUDA-event time is a separate diagnostic and is never substituted into that ratio.

[Device qualification](demo/README.md#device-qualification) is a separate operation using `demo/qualify_u50.py`. It compares Accelerator Monitor (AM) device-active timing with a matched CUDA-graph GPU measurement method. Do not combine its results with notebook host-envelope timings.

These guides define commands and required checks, not a new board PASS or performance result. Use reports from the artifact actually built and executed; historical release measurements do not qualify a changed implementation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `source/` | HLS kernel, responsibility headers, HLS/connectivity configuration and physical Tcl hooks. |
| `verification/` | Independent public-ABI testbench, co-simulation profiles and generated-adapter checks. |
| `demo/` | Notebook, local runtime, headless notebook runner and separate device qualification. |
| `spec/` | Public callable, types, algorithms, ownership rules and figures. |
| `Makefile` | Build, RTL co-simulation, implementation and actual notebook execution entry points. |
| `requirements.md` | Environment, supported local models, kernel ABI and measurement boundaries. |

Licensed under Apache-2.0; see [LICENSE](LICENSE).
