#!/usr/bin/env python3
"""Execute the live notebook and its real hardware callback without a server.

The source notebook is unchanged. An executed copy, artifact identity and the
callback result are saved under --output, including notebook outputs on failure.
This is the notebook's host-envelope demonstration, not device-only qualification.
All scalable clocks must match routed timing through the shared clock gate;
its narrowly reviewed driver-specific HBM quantization is recorded, not hidden
by a wider clock tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from qualify_u50 import clock_contract, routed_timing_report, section


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observer_cell(artifact: dict[str, str], result_path: Path, clock_binding: dict) -> str:
    return ("_expected_artifact = " + repr(artifact) + "\n_result_path = " + repr(str(result_path))
            + "\n_clock_binding = " + repr(clock_binding) + "\n") + '''
import hashlib
import json
import math
from pathlib import Path
from qualify_u50 import check_clock as _notebook_check_clock, clock_contract as _notebook_clock_contract

def _notebook_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

assert session.config.xclbin == Path(_expected_artifact["xclbin"])
assert _notebook_hash(session.config.xclbin) == _expected_artifact["sha256"]
assert _notebook_hash(_clock_binding["report"]) == _clock_binding["report_sha256"]
_expected_uuid = session._u50._xrt.xclbin(str(session.config.xclbin)).get_uuid().to_string().lower()
assert session._u50._device.get_xclbin_uuid().to_string().lower() == _expected_uuid
assert _notebook_clock_contract({"clocks": _clock_binding["clocks"]},
                               _clock_binding["topology"]) == _clock_binding["clocks"]

_bdf = session._u50._device.get_info(session._u50._xrt.xrt_info_device.bdf)
if _bdf.count(":") == 1:
    _bdf = "0000:" + _bdf
_pci = Path("/sys/bus/pci/devices") / _bdf.lower()
assert _pci.is_dir(), "cannot bind notebook device to PCI sysfs"
_clock_before = _notebook_check_clock(_pci, _clock_binding["clocks"])

_captured_results = []
_actual_benchmark = session.benchmark
def _capture_benchmark(prompt):
    result = _actual_benchmark(prompt)
    _captured_results.append(result)
    return result

session.benchmark = _capture_benchmark
try:
    on_run(run_button)
finally:
    session.benchmark = _actual_benchmark

_clock_after = _notebook_check_clock(_pci, _clock_binding["clocks"])
assert len(_captured_results) == 1, "the actual notebook benchmark was not executed exactly once"
_live_result = _captured_results[0]
assert _live_result["correctness_passed"] is True
assert math.isfinite(_live_result["max_absolute_error"]) and _live_result["max_absolute_error"] >= 0
assert 1 <= _live_result["prompt_tokens"] <= 8
for _key in ("pure_rtx_host_invocation_us", "hybrid_u50_host_invocation_us",
             "host_invocation_ratio", "pure_rtx_cuda_event_us"):
    assert math.isfinite(_live_result[_key]) and _live_result[_key] > 0, _key
assert _live_result["measurement_note"]
assert _notebook_hash(session.config.xclbin) == _expected_artifact["sha256"]
assert _notebook_hash(_clock_binding["report"]) == _clock_binding["report_sha256"]
_loaded_uuid = session._u50._device.get_xclbin_uuid().to_string().lower()
assert _loaded_uuid == _expected_uuid, "loaded image changed during notebook execution"

_record = {"status": "PASS", "artifact": dict(_expected_artifact, loaded_uuid=_loaded_uuid),
           "clock": {"contract": _clock_binding, "pci_path": str(_pci),
                     "before": _clock_before, "after": _clock_after}, "result": _live_result}
Path(_result_path).write_text(json.dumps(_record, indent=2, allow_nan=False) + "\\n")
_rows = [
    ("Actual hardware callbacks", len(_captured_results)),
    ("Prompt tokens", _live_result["prompt_tokens"]),
    ("RTX host invocation (us)", _live_result["pure_rtx_host_invocation_us"]),
    ("U50 host invocation (us)", _live_result["hybrid_u50_host_invocation_us"]),
    ("Host invocation ratio (RTX / U50)", _live_result["host_invocation_ratio"]),
    ("RTX CUDA-event diagnostic (us; not in ratio)", _live_result["pure_rtx_cuda_event_us"]),
    ("Maximum absolute error", _live_result["max_absolute_error"]),
    ("Correctness", "finite AND (absolute <= 0.0625 OR relative <= 0.08); reference floor 1e-6"),
    ("Xclbin SHA256", _expected_artifact["sha256"]),
    ("Loaded UUID", _loaded_uuid),
    ("Routed clock report SHA256", _clock_binding["report_sha256"]),
    ("Physical clocks before callback (MHz)", _clock_before["measured_mhz"]),
    ("Physical clocks after callback (MHz)", _clock_after["measured_mhz"]),
]
for _kind, _domain in _clock_binding["clocks"].items():
    _rows.append((f"{_kind} routed / metadata / effective clock (MHz)",
                  f"{_domain['frequency_mhz']} / {_domain['artifact_frequency_mhz']} / "
                  f"{_domain['effective_frequency_mhz']}"))
_rows.append(("Management clock driver profile", _clock_binding["clocks"]["SYSTEM"]["driver_version"]))
_rows.append(("HBM clock capability", _clock_binding["clocks"]["SYSTEM"]["capability_reason"]))
_lines = ["| Check | Actual result |", "| --- | --- |"]
_lines.extend(f"| {name} | {value} |" for name, value in _rows)
display(Markdown("### Notebook hardware callback: PASS\\n\\n" + "\\n".join(_lines)))
display(Markdown(_live_result["measurement_note"]))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--routed-clock-report", type=Path,
                        help="routed timing report; defaults to ROUTED_CLOCK_REPORT or the artifact build layout")
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.xclbin = args.xclbin.expanduser().resolve()
    if not args.model_dir.is_dir() or not args.xclbin.is_file() or args.device < 0:
        parser.error("require an existing model directory, explicit xclbin file and non-negative device")

    demo_dir = Path(__file__).resolve().parent
    notebook = nbformat.read(demo_dir / "demo.ipynb", as_version=4)
    artifact = {"xclbin": str(args.xclbin), "sha256": sha256(args.xclbin)}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    try:
        report = (args.routed_clock_report or os.environ.get("ROUTED_CLOCK_REPORT")
                  or args.xclbin.parent / "report/hw/link/imp/impl_1_hw_bb_locked_timing_summary_post_route_physopt.rpt")
        report = Path(report).expanduser().resolve(strict=True)
        routed = routed_timing_report(report)
        topology = section(args.xclbin, "CLOCK_FREQ_TOPOLOGY", output)
        contract = clock_contract(routed, topology)
        binding = {"report": str(report), "report_sha256": routed["report_sha256"],
                   "clocks": contract, "topology": topology}
        notebook.cells.append(nbformat.v4.new_code_cell(observer_cell(artifact, result_path, binding)))
        os.environ.update(FPT26_MODEL_DIR=str(args.model_dir), FPT26_XCLBIN=str(args.xclbin),
                          FPT26_DEVICE=str(args.device), NVIDIA_TF32_OVERRIDE="0")
        if sha256(args.xclbin) != artifact["sha256"] or sha256(report) != binding["report_sha256"]:
            raise RuntimeError("clock-contract inputs changed before notebook execution")
        NotebookClient(notebook, kernel_name="python3", timeout=1800,
                       allow_errors=False, store_widget_state=True,
                       record_timing=True).execute(cwd=str(demo_dir))
        if sha256(args.xclbin) != artifact["sha256"]:
            raise RuntimeError("xclbin file changed during notebook execution")
        if sha256(report) != binding["report_sha256"]:
            raise RuntimeError("routed timing report changed during notebook execution")
        result = json.loads(result_path.read_text())
        if result["status"] != "PASS":
            raise RuntimeError("notebook hardware callback did not pass")
    except BaseException as error:
        result_path.write_text(json.dumps({"status": "FAIL", "artifact": artifact,
                               "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise
    finally:
        nbformat.write(notebook, output / "executed.ipynb")
    print(f"PASS: {output / 'executed.ipynb'}")


if __name__ == "__main__":
    main()
