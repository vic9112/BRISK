# Public-ABI verification

[brisk_hls_tb.cpp](brisk_hls_tb.cpp) derives inputs and expected results from the [public specification](../spec/README.md) and [public declarations](../spec/brisk_types.hpp). The oracle does not include DUT implementation headers, call DUT arithmetic helpers or use DUT output as a golden result. Build configuration supplies the implementation separately.

Run from the repository root:

```bash
make hls
make hls-cosim
```

`make hls` includes C simulation and synthesis. `make hls-cosim` runs the complete RTL case-group union. Setup or test-vector generation alone is not RTL completion; every selected group must finish its actual RTL transactions and checks.

## Suites and expected coverage

| Testbench selection | Meaning |
| --- | --- |
| `--suite numeric` or `--suite cosim` | Bounded numerical/integration group; `cosim` is the default alias. |
| `--suite paper` | Paper-shaped prefill/decode sequence, continued through live displacement. |
| `--suite residency` | Resident windows, host tiling, overflow and cold nonallocation. |
| `--suite isolation` | Interleaved layers and complete retirement/continuation sequences. |
| `--suite identity` | Signed-identity scenarios and continuations. |
| `--suite validation` | Scalar rejections between a valid prefill and state-preserving continuation. |
| `--suite full` | All groups, using the same independent oracle and case content. |
| Unknown option or suite | Nonzero exit; no silently ignored selection. |

The following table specifies expected testbench work. It is not an execution result or a claim that a partially completed group set is full RTL coverage.

| Group | Successful calls | Rejections | Compared output elements |
| --- | ---: | ---: | ---: |
| `numeric` | 7 | 0 | 69,632 |
| `paper` | 15 | 0 | 180,224 |
| `residency` | 9 | 0 | 385,024 |
| `isolation` | 9 | 0 | 86,016 |
| `identity` | 4 | 0 | 36,864 |
| `validation` | 2 | 22 | 16,384 |
| Complete union / `full` | 46 | 22 | 774,144 |

Each group retains its complete stateful sequence. Selection does not split a continuation chain across reset runs, replace the oracle or omit rejected-call checks.

| Minimum completed work per selected RTL suite | Requirement |
| --- | ---: |
| Successful dataflow invocations | 2 |
| Actual public-ABI transactions | 3 |

## Numerical and memory checks

The numerical oracle exactly expands BF16 inputs, then independently evaluates dense causal FP32 dot products, stable dense softmax with the standard exponential, weighted sums and final division. It rounds only final output to BF16. The dense oracle does not reproduce the DUT's online recurrence. An independent slot-ownership model predicts raw backing changes.

| Check | Required result |
| --- | --- |
| General output | Finite actual/reference and `abs(error) <= max(0.0625,0.08*max(abs(reference),1e-6))`. |
| Exact analytic patterns | Independently calculated BF16 encodings, including ties-to-even and signed zero. |
| Precision discriminators | Distinguish FP32 scores and factors from forbidden BF16 intermediate conversions. |
| C-stage input preservation | Preserve each logical input region, its guards and the maximum unused input transport tails. |
| Active-K/V input in RTL | The native wrapper does not post-read back this read-only port. Its host-side preservation assertion is a C-stage check, not observed absence of RTL writes. |
| Outputs | Check every expected element and preserve output guards. |
| Backing | Compare every logical raw backing line and its guards after each successful call. |
| Invalid calls | Exact first-failing scalar status; outputs, backing and retained context unchanged; a subsequent valid continuation checks state preservation. |
| Nonallocation | Cold history is established before first use; later output/backing checks detect an unauthorized resident fill or early writeback without mutating live history. |

## Co-simulation transport profiles

The functional kernel bounds do not shrink for RTL co-simulation. The test profiles size instrumented memory transport for their complete selected workloads while preserving stable backing pointers, every logical region and adjacent guards. [hls_adapter_config.cfg](hls_adapter_config.cfg) and [hls_identity_config.cfg](hls_identity_config.cfg) define the exact profile depths. [check_adapter_rtl.py](check_adapter_rtl.py) checks generated adapter structure separately from numerical results.

| Transport obligation | Bound or scope |
| --- | --- |
| Normal profile selection | `numeric`, `paper`, `residency`, `isolation` and `validation`. |
| Identity profile selection | `identity`; its maximum-stride allocation requires the wider backing transport. |
| Q/O transport, both profiles | `2,564` words per port, or `82,048` bytes: maximum exercised `1*40*8*8 = 2,560` logical words plus `4` guards. |
| Active-K/V transport, both profiles | `5,124` packs, or `163,968` bytes: maximum exercised `1*40*8*16 = 5,120` logical packs plus `4` guards. |
| Normal-profile backing transport | `1,892` lines per port: maximum exercised `2*59*2*8 = 1,888` logical lines plus `4` guards. |
| Identity-profile backing transport | `131,076` lines per port: maximum exercised `1*8192*2*8 = 131,072` logical lines plus `4` guards. |
| RTL-observed logical regions and guards | Q/O and backing allocations, including their selected logical regions and guards. The read-only `active_kv` input is not post-read back by the native wrapper. |
| Maximum unused public transport tails | Checked in the C stage only; not claimed as serialized or checked by the bounded RTL profiles. |
| Production bounds | Remain those declared by the kernel specification and production HLS configuration. |

Use `make hls-cosim` for the complete RTL union so each group receives its matching transport profile; do not override the normal profile to run the monolithic `full` selection. Guard coverage and bounded adapter depths do not prove the maximum host-sequence workload in RTL. Do not label a maximum legal stride test as a maximum-size transaction. Native wrapper range diagnostics are incomplete and some address/page gaps can be silent; checked buffers and guards are not a universal memory-safety proof.

## Results and limits

The testbench emits named case completion, a first divergence when applicable, count tables and an explicit pass/fail terminator; a failed check or exception produces nonzero exit status. Keep complete logs and reports under the ignored build directory. Read the actual completed group results before claiming the complete union.

C/RTL agreement does not establish arbitrary AXI-stall safety, reset timing, external write visibility, achieved frequency, utilization or board performance. Inspect generated adapters and applicable transaction/waveform evidence, complete routed implementation, then execute the actual notebook and any separate device qualification. This guide reports no new execution PASS.
