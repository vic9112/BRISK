# Local attention demo

The notebook compares prompt-derived attention from a local Llama checkpoint on the RTX GPU and U50. It is not a text-generation benchmark: projections, rotary embedding and output projection are outside the timed attention boundary.

## Execute the notebook

With the [required tools, devices and local model](../requirements.md) available:

```bash
MODEL_DIR=/local/model make demo
```

The Make target builds the default artifact if needed and calls [run_notebook.py](run_notebook.py). The runner executes every code cell and invokes the notebook's actual button callback; it does not replace the callback with a separate benchmark implementation. JupyterLab is not required for this headless path. [demo_runtime.py](demo_runtime.py) supplies the model loading, tensor preparation and hardware calls.

To select an existing artifact explicitly:

```bash
MODEL_DIR=/local/model XCLBIN=/path/to/kernel.hw.xclbin \
  ROUTED_CLOCK_REPORT=/path/to/matching-routed-timing.rpt make demo
```

The explicit file and its matching routed timing report must exist. Neither this path nor the default path downloads a release artifact or model weights. The default artifact is `build/llama3_fast_attention.hw.xclbin`; Make builds it and supplies its normal routed report when needed.

## Clock checks before execution

The runner accepts `--routed-clock-report`; otherwise it uses a supplied `ROUTED_CLOCK_REPORT`, then defaults to `report/hw/link/imp/impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt` relative to the selected xclbin's parent directory. Set `ROUTED_CLOCK_REPORT` when an external artifact's matching report is not at that standard location. Do not substitute a report from another build.

| Check stage | Required condition |
| --- | --- |
| Before notebook startup or image loading | Matching routed timing report and artifact topology for `DATA` (`kernel00`), `KERNEL` (`kernel01`) and the single `SYSTEM` (HBM) clock. |
| After image loading, before the actual callback | Sample physical clocks and check the effective clock contract before running attention. |
| After the actual callback | Sample physical clocks again and retain the checks with the run result. |

Clock targets and metadata are not substitutes for the effective operating clocks. The effective clocks must meet the bound routed contract; a mismatch stops execution rather than silently programming or accepting a different clock configuration.

A raw HBM metadata request may select a different effective setting under the [narrowly supported XRT mapping](../requirements.md#version-bound-hbm-clock-handling). This exception applies only to SYSTEM/HBM, not generic clock rounding. Every DATA, KERNEL and SYSTEM physical counter remains mandatory before kernel execution, and effective clocks must match routed STA. The request mapping is not a claim that physical clocks or a board run have already passed.

| Input or domain | Requirement |
| --- | --- |
| `MODEL_DIR` | Existing supported Hugging Face or official Meta checkpoint; see the environment guide. |
| `DEVICE` | XRT device index; default `0`. |
| Model geometry | `32` query heads, `8` K/V heads and dimension `128`. |
| Active prompt | Batch `1`, from `1` through `8` tokens. |
| Numerical guard | Finite actual/reference values and `abs(O-O_ref) <= max(0.0625,0.08*max(abs(O_ref),1e-6))`. |
| GPU arithmetic | FP32 QK, softmax probabilities and SV; final BF16 conversion only. TF32 is disabled. |
| Actual callback executions | `1` per headless runner invocation. |

A configuration/import check, notebook opening or displayed widget alone is not successful execution. Require completed hardware calls, finite checked outputs and successful completion of the callback path.

The Make target selects a new `build/demo/<timestamp>/` directory for `executed.ipynb` and `result.json`. The runner retains the executed notebook on failure and returns a failing process status; inspect those outputs rather than inferring success from notebook startup. The location follows `BUILD_DIR` when changed.

## Timing scopes

| Notebook field | Scope |
| --- | --- |
| RTX host invocation | Host-observed attention call through CUDA completion. |
| U50 host invocation | Kernel submission through XRT completion wait. |
| Displayed ratio | RTX host invocation divided by U50 host invocation. |
| CUDA-event diagnostic | GPU event timing, displayed separately and never used in the host-envelope ratio. |

Use the same prompt-derived post-RoPE BF16 Q/K/V boundary for both devices. Do not describe the notebook ratio as device-active speedup or full-model throughput.

## Device qualification

[qualify_u50.py](qualify_u50.py) runs the separate rigorous device comparison. The Make entry supplies the artifact and routed-clock evidence; this path currently requires official Meta checkpoint material and its sibling `llama` package:

```bash
MODEL_DIR=/local/meta-model make qualify
```

For direct invocation, inspect the required artifact, routed-clock and report-directory arguments:

```bash
python3 demo/qualify_u50.py --help
```

| Qualification setting | Current procedure |
| --- | --- |
| Batch / stride | `2` / `32`. |
| Prefill | `8` tokens with no earlier history. |
| Successive decode | Earlier-token counts from `8` through `23`; report paper-shaped, capacity-boundary and overflow work separately. |
| Resident positions per batch | `20`, including current data. |
| Default warm-up / measured series | `2` / `5`; all outputs, including warm-up outputs, are checked. |
| Minimum warm-up / measured series | `1` / `3`. |
| FPGA time | Accelerator Monitor (AM) execution cycles divided by the routed clock, with physical clock and monitor checks; no host-timing fallback. |
| GPU time | CUDA-graph replay of unfused FP32 dense operators, using captured internal CUDA-event timing. TF32 is disabled. |
| Numerical reference | Independent NumPy FP32 dense causal attention from BF16 inputs, rounded to BF16 at output. |
| Numerical guard | Finite outputs and `abs(error) <= max(0.0625,0.08*max(abs(reference),1e-6))`. |

The compared boundary includes K/V state update and attention output before projection. Input preparation, transfers and host scheduling are excluded. Device-side graph scheduling and inter-operator overhead remain included on the GPU. This is not an optimized SDPA, FlashAttention or state-of-the-art GPU comparison.

AM device-active timing and the matched CUDA-graph procedure are separate from notebook host timing. Preserve the method, workload, warm-up/sample policy, actual artifact and complete numerical checks with any reported comparison. CUDA-event diagnostics are not interchangeable with AM device-active measurements or host invocation envelopes. The Make target selects a new `build/qualification/<timestamp>/` report directory containing `qualification.md`, `result.json` and detailed checks. The location follows `BUILD_DIR` when changed. Direct invocation uses `--report-dir`, not the notebook runner's `--output` option.

For an explicitly selected artifact, provide its matching `ROUTED_CLOCK_REPORT` and `KERNEL_FREQ_MHZ`; do not reuse timing evidence from another build. See the [source guide](../source/README.md#configuration-ownership-and-checks) for the default report location.

No historical release identity, timing result or current board PASS is asserted here. Reports must come from the artifact and local model actually executed.
