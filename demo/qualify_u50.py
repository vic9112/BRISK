#!/usr/bin/env python3
"""BRISK attention-only device qualification using an explicitly selected image.

This is not full-model generation, an optimized GPU/SOTA benchmark, or a
replacement for independent C/RTL verification. No DUT helper is imported.
Inputs come from official Meta layer-zero weights and RoPE helpers; the CPU
oracle independently evaluates dense attention in NumPy FP32. The separately
timed GPU baseline replays CUDA graphs of unfused PyTorch FP32 operations,
with internal external-event timing nodes and TF32 disabled.

Clock preflight binds all scalable domains to routed timing. The only raw
metadata exception is the reviewed XRT 2.18.179 HBM table bucket selecting
400 MHz for integer requests from 400 through 404; physical readback is required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time


KERNEL = "llama3_fast_attention"
CU = KERNEL + "_1"
BATCH, HEADS, KV_HEADS, DIM, LANES = 2, 32, 8, 128, 4
STRIDE, RESIDENT_PER_BATCH = 32, 20
TOKENS = 24
CALLS = [(0, 8)] + [(position, 1) for position in range(8, TOKENS)]
CLOCK_DOMAINS = {
    "DATA": ("clk_kernel_00_unbuffered_net", 0),
    "KERNEL": ("clk_kernel_01_unbuffered_net", 1),
    "SYSTEM": ("hbm_aclk", 2),
}
AM_FIELDS = (
    "end_count", "start_count", "exec_cycles", "stall_int_cycles",
    "stall_str_cycles", "stall_ext_cycles", "busy_cycles",
    "max_parallel_iterations", "max_exec_cycles", "min_exec_cycles",
)
AIM_FIELDS = (
    "write_bytes", "write_transactions", "write_latency", "write_busy_cycles",
    "read_bytes", "read_transactions", "read_latency", "read_busy_cycles",
    "outstanding_count", "last_write_address", "last_write_data",
    "last_read_address", "last_read_data",
)
PROMPTS = (
    "A careful hardware engineer checks every attention head, every earlier "
    "token, and every batch independently before measuring latency. The river "
    "flows past a quiet mountain village while the morning sunlight warms the "
    "stone bridge and the old wooden houses.",
    "An astronomer records the changing night sky and compares each observation "
    "with an independent mathematical prediction. A small spacecraft crosses "
    "the distant solar system carrying instruments, maps, and a message about "
    "the blue oceans and green forests of Earth.",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def routed_timing_report(path, clock_mhz=None):
    raw = path.read_bytes()
    report = raw.decode("utf-8")
    states = re.findall(r"(?m)^\| Design State[ \t]*:[ \t]*([^\r\n]+)", report)
    require([state.strip() for state in states] == ["Routed"], "timing report is not uniquely Routed")

    def section(name, following):
        starts = list(re.finditer(r"(?m)^\| " + re.escape(name) + r"[ \t]*$", report))
        ends = list(re.finditer(r"(?m)^\| " + re.escape(following) + r"[ \t]*$", report))
        require(len(starts) == len(ends) == 1 and starts[0].end() < ends[0].start(),
                f"missing, duplicate or unordered timing-report section: {name}")
        return report[starts[0].end():ends[0].start()]

    summary = section("Design Timing Summary", "Clock Summary")
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    header = ("WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints "
              "WHS(ns) THS(ns) THS Failing Endpoints THS Total Endpoints "
              "WPWS(ns) TPWS(ns) TPWS Failing Endpoints TPWS Total Endpoints").split()
    headers = [index for index, line in enumerate(lines) if line.split() == header]
    require(len(headers) == 1 and headers[0] + 2 < len(lines), "missing or duplicate design timing row")
    words = lines[headers[0] + 2].split()
    require(len(words) == 12, "unexpected design timing row format")
    values = [float(word) if index % 4 < 2 else int(word) for index, word in enumerate(words)]
    for offset, name in ((0, "setup"), (4, "hold"), (8, "pulse width")):
        worst, total, failing, endpoints = values[offset:offset + 4]
        require(math.isfinite(worst) and math.isfinite(total) and worst >= 0 and total == 0
                and failing == 0 and endpoints > 0, f"routed {name} timing is open or unmeasured")
    require("All user specified timing constraints are met." in lines,
            "timing report does not declare all constraints met")

    clock_text = section("Clock Summary", "Intra Clock Table")
    rounding = 0.0005 + 1e-9  # Both report columns print three decimal places.
    clocks = {}
    for kind, (clock_name, index) in CLOCK_DOMAINS.items():
        rows = [line.split() for line in clock_text.splitlines()
                if line.split() and line.split()[0] == clock_name]
        require(len(rows) == 1 and len(rows[0]) == 5 and rows[0][1].startswith("{")
                and rows[0][2].endswith("}"), f"missing, duplicate or malformed {kind} clock row")
        period, frequency = map(float, rows[0][3:])
        require(math.isfinite(period) and math.isfinite(frequency) and period > 0 and frequency > 0
                and abs(period - 1000.0 / frequency) <= rounding,
                f"inconsistent routed {kind} clock period/frequency")
        clocks[kind] = {"name": clock_name, "counter_index": index,
                        "period_ns": period, "frequency_mhz": frequency}
    data = clocks["DATA"]
    if clock_mhz is not None:
        require(math.isfinite(clock_mhz) and clock_mhz > 0
                and abs(data["period_ns"] - 1000.0 / clock_mhz) <= rounding
                and abs(data["frequency_mhz"] - clock_mhz) <= rounding,
                "routed kernel clock differs from requested clock")
    names = ("wns_ns", "tns_ns", "setup_failing_endpoints", "setup_total_endpoints",
             "whs_ns", "ths_ns", "hold_failing_endpoints", "hold_total_endpoints",
             "wpws_ns", "tpws_ns", "pulse_failing_endpoints", "pulse_total_endpoints")
    return {"report_sha256": hashlib.sha256(raw).hexdigest(), "design_state": "Routed",
            "clock_name": data["name"], "period_ns": data["period_ns"],
            "frequency_mhz": data["frequency_mhz"], "clocks": clocks,
            **dict(zip(names, values))}


def clock_contract(routed_timing, topology):
    entries = topology["m_clock_freq"]
    require(int(topology["m_count"]) == len(entries) == len(CLOCK_DOMAINS),
            "expected exactly DATA, KERNEL and one SYSTEM clock")
    driver_version = Path("/sys/module/xclmgmt/version").read_text().strip()
    driver_profile = driver_version.split(",", 1)[0]
    contract = {}
    for kind in CLOCK_DOMAINS:
        matches = [entry for entry in entries if entry["m_type"] == kind]
        require(len(matches) == 1, f"missing, duplicate or ambiguous {kind} clock metadata")
        frequency = float(matches[0]["m_freq_Mhz"])
        routed = routed_timing["clocks"][kind]
        require(math.isfinite(frequency) and frequency.is_integer() and 0 < frequency <= 65535,
                f"artifact {kind} clock is not an unsigned integer MHz request")
        effective = frequency
        reason = "raw metadata matches routed target"
        if frequency != routed["frequency_mhz"]:
            # This reviewed non-Versal driver floors between its adjacent
            # 400/405-MHz entries. It is not a generic step or wider tolerance.
            require(kind == "SYSTEM" and routed["frequency_mhz"] == 400
                    and 400 <= frequency < 405 and driver_profile == "2.18.179",
                    f"unsupported artifact {kind} clock request/driver profile")
            effective = 400.0
            reason = "XRT 2.18.179 UltraScale HBM table floor: [400,405) MHz selects 400 MHz"
        require(abs(effective - routed["frequency_mhz"]) <= 0.0005 + 1e-9,
                f"artifact {kind} clock differs from routed timing")
        contract[kind] = {**routed, "artifact_name": matches[0].get("m_name", ""),
                          "artifact_frequency_mhz": frequency, "effective_frequency_mhz": effective,
                          "driver_profile": driver_profile, "driver_version": driver_version,
                          "capability_reason": reason}
    return contract


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bf16_float(words, np):
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_round(values, np):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & 1)
    return (rounded >> np.uint32(16)).astype("<u2")


def compare(actual_words, reference_words, np):
    actual, reference = bf16_float(actual_words, np), bf16_float(reference_words, np)
    finite = np.isfinite(actual) & np.isfinite(reference)
    absolute = np.abs(actual - reference)
    relative = absolute / np.maximum(np.abs(reference), np.float32(1e-6))
    passed = finite & ((absolute <= 0.0625) | (relative <= 0.08))
    return {
        "passed": bool(passed.all()), "elements": int(passed.size),
        "nonfinite": int((~finite).sum()), "failed_elements": int((~passed).sum()),
        "max_absolute_error": float(absolute.max()) if finite.all() else None,
        "max_relative_error": float(relative.max()) if finite.all() else None,
    }


def model_inputs(model_dir, output, torch, np, gpu):
    """Use only selected weights; never instantiate official Attention/Transformer."""
    package = model_dir.parent / "llama"
    model = load_source("brisk_official_meta_model", package / "model.py")
    tokenizer_module = load_source("brisk_official_meta_tokenizer", package / "tokenizer.py")
    params = json.loads((model_dir / "params.json").read_text())
    require((params["dim"], params["n_heads"], params["n_kv_heads"], params["n_layers"])
            == (4096, HEADS, KV_HEADS, 32), "unexpected official model geometry")
    tokenizer = tokenizer_module.Tokenizer(str(model_dir / "tokenizer.model"))
    encoded = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in PROMPTS]
    require(all(len(tokens) >= TOKENS for tokens in encoded), "prompt too short")
    token_ids = np.asarray([tokens[:TOKENS] for tokens in encoded], dtype=np.int64)
    require(not np.array_equal(token_ids[0], token_ids[1]), "batches must be distinct")
    checkpoint = model_dir / "consolidated.00.pth"
    # mmap keeps unrelated layers disk-backed; only embeddings for selected IDs,
    # attention norm, and layer-zero Q/K/V projections are copied to the GPU.
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    names = ["layers.0.attention_norm.weight"] + [
        f"layers.0.attention.w{projection}.weight" for projection in "qkv"
    ]
    weight_hashes = {
        name: hashlib.sha256(state[name].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name in names
    }
    ids = torch.from_numpy(token_ids)
    embeddings = state["tok_embeddings.weight"][ids].contiguous()
    weight_hashes["selected_token_embeddings"] = hashlib.sha256(
        embeddings.view(torch.uint8).numpy().tobytes()).hexdigest()
    hidden = embeddings.to(device=gpu, dtype=torch.bfloat16)
    norm = model.RMSNorm(params["dim"], eps=params["norm_eps"]).to(
        device=gpu, dtype=torch.bfloat16)
    norm.weight.copy_(state[names[0]].to(device=gpu, dtype=torch.bfloat16))
    hidden = norm(hidden)
    projected = []
    for projection, heads in (("q", HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
        weight = state[f"layers.0.attention.w{projection}.weight"].to(
            device=gpu, dtype=torch.bfloat16)
        projected.append(torch.nn.functional.linear(hidden, weight).reshape(
            BATCH, TOKENS, heads, DIM))
    frequencies = model.precompute_freqs_cis(DIM, TOKENS, params["rope_theta"]).to(gpu)
    q, k = model.apply_rotary_emb(projected[0], projected[1], frequencies)
    qkv = [tensor.to(torch.bfloat16).contiguous().cpu() for tensor in (q, k, projected[2])]
    require(all(bool(torch.isfinite(tensor.float()).all()) for tensor in qkv),
            "model-derived input is nonfinite")
    words = [tensor.view(torch.uint16).numpy().astype("<u2", copy=True) for tensor in qkv]
    np.savez(output / "inputs.npz", token_ids=token_ids, q=words[0], k=words[1], v=words[2])
    provenance = {
        "boundary": "official layer-zero RMSNorm/QKV projection, official Q/K RoPE, BF16 inputs",
        "not_claimed": "full Llama generation or multi-layer board qualification",
        "model_directory": str(model_dir), "params": params, "prompts": PROMPTS,
        "token_ids": token_ids.tolist(), "selected_weight_sha256": weight_hashes,
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "official_source_sha256": {name: sha256(package / name) for name in ("model.py", "tokenizer.py")},
        "params_sha256": sha256(model_dir / "params.json"),
        "tokenizer_sha256": sha256(model_dir / "tokenizer.model"),
        "inputs_sha256": sha256(output / "inputs.npz"),
    }
    save_json(output / "input-provenance.json", provenance)
    del state, norm, hidden, projected, weight, q, k, embeddings, qkv
    torch.cuda.synchronize(gpu)
    return words


def cpu_reference(qkv, np):
    """Dense FP32 matrix reference, independent of the DUT's online recurrence."""
    q, k, v = [bf16_float(words, np) for words in qkv]
    result = {}
    scale = np.float32(1.0 / math.sqrt(DIM))
    for position, sequence in CALLS:
        out = np.empty((BATCH, sequence, HEADS, DIM), dtype=np.float32)
        for batch in range(BATCH):
            for row in range(sequence):
                end = position + row + 1
                for head in range(HEADS):
                    kv_head = head // (HEADS // KV_HEADS)
                    scores = (k[batch, :end, kv_head] @ q[batch, position + row, head]) * scale
                    weights = np.exp(scores - scores.max()).astype(np.float32)
                    probabilities = weights / weights.sum(dtype=np.float32)
                    out[batch, row, head] = probabilities @ v[batch, :end, kv_head]
        require(bool(np.isfinite(out).all()), "independent reference is nonfinite")
        result[position] = bf16_round(out, np)
    return result


def section(xclbin, name, output):
    destination = output / f"{name.lower()}.json"
    command = ["xclbinutil", "--input", str(xclbin),
               "--dump-section", f"{name}:JSON:{destination}"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    (output / f"{name.lower()}.log").write_text(completed.stdout + completed.stderr)
    return json.loads(destination.read_text())[name.lower()]


def counters(path, fields):
    raw = [int(value) for value in path.read_text().split()]
    require(len(raw) == len(fields), f"unexpected monitor format at {path}")
    return dict(zip(fields, raw))


def delta(before, after, name):
    require(after[name] >= before[name], f"counter reset/wrap detected: {name}")
    return after[name] - before[name]


class Monitors:
    def __init__(self, pci, layout):
        by_base = {}
        for child in pci.iterdir():
            if child.name.startswith(("accel_mon.u.", "aximm_mon.u.")):
                name = (child / "name").read_text().strip()
                base = int(name.rsplit("_", 1)[1])
                require(base not in by_base, "duplicate monitor base address")
                by_base[base] = child / "counters"
        self.am = None
        self.aim = {}
        self.mapping = []
        for entry in layout["m_debug_ip_data"]:
            kind, name = entry["m_type"], entry["m_name"]
            base = int(str(entry["m_base_address"]), 0)
            belongs_to_cu = CU in re.split(r"[/:\s]+", name)
            if kind == "ACCEL_MONITOR" and belongs_to_cu:
                require(self.am is None, "multiple matching accelerator monitors")
                self.am = by_base.get(base)
                require(self.am is not None, "AM missing from selected device sysfs")
            elif kind == "AXI_MM_MONITOR" and belongs_to_cu:
                match = re.search(r"/m_axi_gmem_kv([0-3])(?:[-/]|$)", name)
                if not match:
                    continue
                lane = int(match[1])
                require(lane not in self.aim, "duplicate cache AIM")
                require(base in by_base, f"cache AIM missing from sysfs: {name}")
                self.aim[lane] = by_base[base]
            else:
                continue
            self.mapping.append({"name": name, "type": kind, "base": base,
                                 "counter_path": str(by_base[base])})
        require(self.am is not None and set(self.aim) == set(range(LANES)),
                "AM and every cache-lane AIM are mandatory; no host-timing fallback")

    def snapshot(self):
        return {"am": counters(self.am, AM_FIELDS),
                "aim": {str(lane): counters(self.aim[lane], AIM_FIELDS) for lane in range(LANES)}}


def check_clock(pci, contract):
    management = pci.with_name(pci.name.rsplit(".", 1)[0] + ".0")
    paths = list(management.glob("clock_freq_counter.m.*/clock_counter_freqs"))
    require(len(paths) == 1, "expected selected device's direct physical clock counter")
    values = [int(value) for value in paths[0].read_text().split()]
    require(set(contract) == set(CLOCK_DOMAINS), "incomplete physical clock contract")
    for kind, (_, index) in CLOCK_DOMAINS.items():
        require(len(values) > index and abs(values[index] - contract[kind]["frequency_mhz"]) <= 1.0,
                f"physical {kind} clock differs from routed clock: {values}")
    return {"path": str(paths[0]), "measured_mhz": values,
            "expected_mhz": {kind: clock["frequency_mhz"] for kind, clock in contract.items()},
            "kernel_index": 0, "rounding_tolerance_mhz": 1.0}


class U50:
    def __init__(self, args, debug_layout, np, contract):
        import pyxrt

        self.xrt, self.np = pyxrt, np
        self.device = pyxrt.device(args.device)
        bdf = self.device.get_info(pyxrt.xrt_info_device.bdf)
        if bdf.count(":") == 1:
            bdf = "0000:" + bdf
        self.pci = Path("/sys/bus/pci/devices") / bdf.lower()
        require(self.pci.is_dir(), "cannot bind selected XRT device to PCI sysfs")
        status = (self.pci / "kdsstat").read_text()
        require(re.search(r"outstanding execs:\s*0(?:\s|$)", status) is not None,
                "selected FPGA has outstanding commands")
        require(re.search(r"contexts:\s*0(?:\s|$)", status) is not None,
                "selected FPGA is in use; will not replace another context")
        image = pyxrt.xclbin(str(args.xclbin))
        kernels = [kernel for kernel in image.get_kernels() if kernel.get_name() == KERNEL]
        require(len(kernels) == 1 and kernels[0].get_num_args() == 22, "unexpected kernel ABI")
        self.uuid = self.device.load_xclbin(str(args.xclbin)).to_string()
        require(self.uuid.lower() == image.get_uuid().to_string().lower(), "loaded UUID mismatch")
        self.kernel = pyxrt.kernel(self.device, image.get_uuid(), f"{KERNEL}:{{{CU}}}")
        self.monitors = Monitors(self.pci, debug_layout)
        self.clock_contract = contract
        self.clock_before = check_clock(self.pci, contract)
        self.clock_mhz = args.routed_clock_mhz
        max_lane_bytes = BATCH * 8 * (HEADS // LANES) * DIM * 2
        self.queries = [self.buffer(max_lane_bytes, lane) for lane in range(LANES)]
        self.outputs = [self.buffer(max_lane_bytes, 9 + lane) for lane in range(LANES)]
        self.active = self.buffer(BATCH * 8 * KV_HEADS * DIM * 2 * 2, 4)
        cache_bytes = BATCH * STRIDE * (KV_HEADS // LANES) * 8 * 64
        self.backing = [self.buffer(cache_bytes, 5 + lane) for lane in range(LANES)]
        self.status = self.buffer(4, 21)
        # The only host backing writes are this one-time initialization. No
        # preupdate/write-through, per-call refill, or backing BO reallocation.
        for buffer in self.backing:
            self.to_device(buffer, bytes(cache_bytes))
        self.identity = None
        self.next_position = 0

    def buffer(self, size, argument):
        return self.xrt.bo(self.device, size, self.xrt.bo.flags.normal,
                           self.kernel.group_id(argument))

    def to_device(self, buffer, payload):
        buffer.write(payload, 0)
        buffer.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(payload), 0)

    def from_device(self, buffer, size):
        buffer.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
        return bytes(buffer.read(size, 0))

    def call(self, qkv, position, sequence, epoch):
        np = self.np
        q, k, v = [words[:, position:position + sequence] for words in qkv]
        queries = [np.ascontiguousarray(q[:, :, lane * 8:(lane + 1) * 8]).tobytes()
                   for lane in range(LANES)]
        paired = np.empty((BATCH, sequence, KV_HEADS, DIM // 8, 16), dtype="<u2")
        paired[..., :8] = k.reshape(BATCH, sequence, KV_HEADS, DIM // 8, 8)
        paired[..., 8:] = v.reshape(BATCH, sequence, KV_HEADS, DIM // 8, 8)
        active = paired.tobytes()
        identity = epoch << 5  # Layer zero; epoch stays fixed within this series.
        if position == 0:
            self.identity, self.next_position = identity, 0
        require(self.identity == identity and self.next_position == position,
                "demo attempted a replay, gap, or identity change in a live sequence")
        started = time.perf_counter_ns()
        for buffer, payload in zip(self.queries, queries):
            self.to_device(buffer, payload)
        self.to_device(self.active, active)
        self.to_device(self.status, (-1).to_bytes(4, "little", signed=True))
        h2d_us = (time.perf_counter_ns() - started) / 1000
        self.last_snapshot = {"position": position, "sequence": sequence}
        clock_before = check_clock(self.pci, self.clock_contract)
        self.last_snapshot["clock_before"] = clock_before
        before = self.monitors.snapshot()
        self.last_snapshot["before"] = before
        arguments = [*self.queries, self.active, *self.backing, *self.outputs,
                     BATCH, sequence, position, STRIDE, position, 1, 0, identity, self.status]
        require(len(arguments) == 22, "internal ABI construction error")
        started = time.perf_counter_ns()
        run = self.kernel(*arguments)
        # Request a wait timeout, not a hard wall-clock bound or hardware abort.
        state = run.wait(10_000)
        host_us = (time.perf_counter_ns() - started) / 1000
        self.last_snapshot.update(wait_state=str(state), host_invoke_us=host_us)
        require(state == self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED,
                f"XRT run did not complete: {state}; timeout does not abort; no automatic abort/reset")
        after = self.monitors.snapshot()
        self.last_snapshot["after"] = after
        clock_after = check_clock(self.pci, self.clock_contract)
        self.last_snapshot["clock_after"] = clock_after
        require(delta(before["am"], after["am"], "end_count") == 1,
                "AM did not observe exactly one completed execution")
        require(delta(before["am"], after["am"], "start_count") == 1,
                "AM start-edge multiplicity is ambiguous; no latency claim")
        cycles = delta(before["am"], after["am"], "exec_cycles")
        require(cycles > 0, "AM execution-cycle delta is zero")
        traffic = {}
        bytes_per_position_lane = BATCH * (KV_HEADS // LANES) * DIM * 2 * 2
        expected_read = max(0, position - RESIDENT_PER_BATCH) * bytes_per_position_lane
        expected_write = (max(0, position + sequence - RESIDENT_PER_BATCH)
                          - max(0, position - RESIDENT_PER_BATCH)) * bytes_per_position_lane
        for lane in range(LANES):
            key = str(lane)
            traffic[key] = {name: delta(before["aim"][key], after["aim"][key], name)
                            for name in ("read_bytes", "write_bytes", "read_transactions", "write_transactions")}
        started = time.perf_counter_ns()
        status = int.from_bytes(self.from_device(self.status, 4), "little", signed=True)
        out = np.empty((BATCH, sequence, HEADS, DIM), dtype="<u2")
        lane_bytes = len(queries[0])
        for lane, buffer in enumerate(self.outputs):
            words = np.frombuffer(self.from_device(buffer, lane_bytes), dtype="<u2")
            out[:, :, lane * 8:(lane + 1) * 8] = words.reshape(BATCH, sequence, 8, DIM)
        d2h_us = (time.perf_counter_ns() - started) / 1000
        require(status == 0, f"kernel status {status}")
        self.next_position = position + sequence
        return out, {
            "u50_cycles": cycles, "u50_device_us": cycles / self.clock_mhz,
            "u50_host_invoke_us": host_us, "u50_h2d_host_us": h2d_us,
            "u50_d2h_host_us": d2h_us, "u50_h2d_bytes": sum(map(len, queries)) + len(active) + 4,
            "u50_d2h_bytes": LANES * lane_bytes + 4,
            "am_before": before["am"], "am_after": after["am"],
            "aim_before": before["aim"], "aim_after": after["aim"],
            "clock_before": clock_before, "clock_after": clock_after,
            "cache_traffic": traffic, "expected_read_bytes_per_lane": expected_read,
            "expected_write_bytes_per_lane": expected_write,
            "traffic_passed": all(value["read_bytes"] == expected_read and
                                  value["write_bytes"] == expected_write for value in traffic.values()),
        }

    def verify_writebacks(self, qkv, series, output):
        """Outside timing, check every evicted BF16 payload bit in HBM."""
        np = self.np
        evicted = TOKENS - RESIDENT_PER_BATCH
        snapshots, checks = {}, []
        cache_bytes = BATCH * STRIDE * (KV_HEADS // LANES) * 8 * 64
        for lane, buffer in enumerate(self.backing):
            words = np.frombuffer(self.from_device(buffer, cache_bytes), dtype="<u2").reshape(
                BATCH, STRIDE, KV_HEADS // LANES, DIM // 8, 16).copy()
            expected = np.empty((BATCH, evicted, KV_HEADS // LANES, DIM // 8, 16), dtype="<u2")
            for index, payload in enumerate(qkv[1:]):
                expected[..., index * 8:(index + 1) * 8] = payload[
                    :, :evicted, lane * 2:(lane + 1) * 2].reshape(BATCH, evicted, 2, DIM // 8, 8)
            snapshots[f"lane{lane}"] = words
            checks.append({"series": series, "lane": lane, "positions_per_batch": evicted,
                           "bf16_words": int(expected.size),
                           "bitwise_passed": bool(np.array_equal(words[:, :evicted], expected))})
        np.savez(output / f"backing-series{series}.npz", **snapshots)
        return checks


class GPU:
    """CUDA-graph replay of unfused FP32 dense operators, not SDPA/SOTA."""
    def __init__(self, qkv, torch, gpu):
        self.torch, self.gpu = torch, gpu
        torch.cuda.synchronize(gpu)
        started = time.perf_counter_ns()
        self.qkv = [torch.from_numpy(words.copy()).view(torch.bfloat16).to(gpu) for words in qkv]
        torch.cuda.synchronize(gpu)
        self.input_h2d_host_us = (time.perf_counter_ns() - started) / 1000
        self.cache_k = torch.zeros((BATCH, STRIDE, KV_HEADS, DIM), dtype=torch.bfloat16, device=gpu)
        self.cache_v = torch.zeros_like(self.cache_k)
        self.cache_k[:, :TOKENS].copy_(self.qkv[1])
        self.cache_v[:, :TOKENS].copy_(self.qkv[2])
        self.output = torch.empty((BATCH, 8, HEADS, DIM), dtype=torch.bfloat16, device=gpu)
        self.masks = {}
        for position, sequence in CALLS:
            rows = torch.arange(sequence, device=gpu)[:, None] + position
            positions = torch.arange(position + sequence, device=gpu)[None, :]
            self.masks[position] = positions > rows
        torch.cuda.synchronize(gpu)
        stream = torch.cuda.Stream(device=gpu)
        stream.wait_stream(torch.cuda.current_stream(gpu))
        with torch.cuda.stream(stream):
            for _ in range(3):
                for position, sequence in CALLS:
                    self.attention(position, sequence)
        stream.synchronize()
        self.graphs = {}
        for position, sequence in CALLS:
            graph = torch.cuda.CUDAGraph()
            start, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
            start.record(stream)
            end.record(stream)
            stream.synchronize()  # Initialize native events before capture, outside timing.
            with torch.cuda.graph(graph, stream=stream):
                start.record()
                self.attention(position, sequence)
                end.record()
            self.graphs[position, sequence] = graph, start, end
        torch.cuda.synchronize(gpu)

    def attention(self, position, sequence):
        torch = self.torch
        q, k, v = [tensor[:, position:position + sequence] for tensor in self.qkv]
        self.cache_k[:, position:position + sequence].copy_(k)
        self.cache_v[:, position:position + sequence].copy_(v)
        # All visible BF16 history is consumed, including after URAM overflow.
        keys = self.cache_k[:, :position + sequence].float().repeat_interleave(4, dim=2)
        values = self.cache_v[:, :position + sequence].float().repeat_interleave(4, dim=2)
        scores = torch.matmul(q.float().transpose(1, 2), keys.transpose(1, 2).transpose(2, 3))
        scores = scores * (1.0 / math.sqrt(DIM))
        scores = scores.masked_fill(self.masks[position][None, None], -float("inf"))
        weights = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        probabilities = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.matmul(probabilities, values.transpose(1, 2)).transpose(1, 2)
        self.output[:, :sequence].copy_(out.to(torch.bfloat16))

    def call(self, position, sequence):
        torch = self.torch
        torch.cuda.synchronize(self.gpu)
        graph, start, end = self.graphs[position, sequence]
        started = time.perf_counter_ns()
        graph.replay()
        end.synchronize()
        host_us = (time.perf_counter_ns() - started) / 1000
        device_us = start.elapsed_time(end) * 1000
        started = time.perf_counter_ns()
        out = self.output[:, :sequence].contiguous().cpu().view(torch.uint16).numpy().copy()
        d2h_us = (time.perf_counter_ns() - started) / 1000
        require(math.isfinite(device_us) and device_us > 0, "invalid CUDA event duration")
        return out, {"gpu_device_us": device_us, "gpu_host_invoke_us": host_us,
                     "gpu_d2h_host_us": d2h_us}


def markdown_report(args, samples, environment):
    lines = ["# Fresh attention-only qualification", "", "PASS", "",
             "The GPU baseline is CUDA Graph replay of unfused FP32 dense operators, not "
             "optimized SDPA, FlashAttention, full-model throughput, or a SOTA comparison. "
             "Internal external-event record nodes exclude host dispatch gaps. Device-side "
             "graph scheduling and inter-operator overhead remain included. The device "
             "comparison does not divide host invocation times or promise historical speedup.", "",
             "Routed report content is checked and its hash retained. The report alone does "
             "not cryptographically bind timing to the xclbin; build provenance requires "
             "separate signoff.", "",
             "Both device boundaries include BF16 K/V state update, dense causal FP32 QK, "
             "standard exp/normalization/SV, and BF16 O before projection. Tokenization, "
             "embedding, RMSNorm/QKV/RoPE input preparation, transfers and host scheduling "
             "are excluded. FPGA time is AM execution cycles divided by the routed clock; "
             "GPU time is the captured internal CUDA-event interval. All qualification-series outputs, "
             "including qualification warmup series, were checked "
             "against an independent NumPy FP32 dense reference rounded to BF16.", "",
             "| Qualification | Value |", "| --- | --- |",
             f"| Xclbin SHA256 | `{args.sha256}` |",
             f"| Loaded UUID | `{environment['loaded_uuid']}` |",
             f"| Routed kernel clock (MHz) | {args.routed_clock_mhz} |",
             "| Routed DATA / KERNEL / SYSTEM clocks (MHz) | " + " / ".join(
                 str(environment["clock_contract"][kind]["frequency_mhz"]) for kind in CLOCK_DOMAINS) + " |",
             *[f"| {kind} raw / effective clock (MHz) | {clock['artifact_frequency_mhz']} / "
               f"{clock['effective_frequency_mhz']} |" for kind, clock in environment["clock_contract"].items()],
             f"| Management clock driver profile | {environment['clock_contract']['SYSTEM']['driver_version']} |",
             f"| HBM clock capability | {environment['clock_contract']['SYSTEM']['capability_reason']} |",
             f"| Routed report SHA256 | `{environment['routed_clock_report_sha256']}` |",
             f"| Routed report period / frequency (ns / MHz) | {environment['routed_timing']['period_ns']} / "
             f"{environment['routed_timing']['frequency_mhz']} |",
             f"| Routed WNS / WHS / WPWS (ns) | {environment['routed_timing']['wns_ns']} / "
             f"{environment['routed_timing']['whs_ns']} / {environment['routed_timing']['wpws_ns']} |",
             f"| Physical clocks before (MHz) | {environment['clock_before']['measured_mhz']} |",
             f"| Physical clocks after (MHz) | {environment['clock_after']['measured_mhz']} |",
             f"| Batch / stride / layer | {BATCH} / {STRIDE} / 0 |",
             f"| Warmup / measured series | {args.warmup} / {args.repeats} |",
             f"| GPU corpus initial H2D host us (outside timing) | {environment['gpu_initial_input_h2d_host_us']:.6f} |",
             "| Absolute OR relative tolerance | 0.0625 OR 0.08 |",
             "| Relative denominator floor | 1e-6 |",
             "| TF32 / reduced-precision probability | disabled / none |", "",
             "Medians below use measured fresh sequences only. Overflow cases are separated "
             "from the paper workload; they do not extend a paper-comparable speedup claim.", "",
             "| Scope | P | S | AM cycles | U50 device us | GPU CUDA us | GPU/U50 | "
             "U50 host invoke us | GPU host invoke us | U50 H2D us | U50 D2H us | GPU D2H us |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for position, sequence in CALLS:
        rows = [sample for sample in samples if sample["measured"] and sample["position"] == position]
        med = lambda key: statistics.median(row[key] for row in rows)
        scope = "paper" if position <= 18 else ("capacity boundary" if position == 19 else "overflow")
        values = [med(key) for key in ("u50_cycles", "u50_device_us", "gpu_device_us")]
        ratio = values[2] / values[1]
        tail = [med(key) for key in ("u50_host_invoke_us", "gpu_host_invoke_us", "u50_h2d_host_us",
                                     "u50_d2h_host_us", "gpu_d2h_host_us")]
        lines.append(f"| {scope} | {position} | {sequence} | " +
                     " | ".join(f"{value:.6f}" for value in [*values, ratio, *tail]) + " |")
    lines += ["", "Cache AIM bytes are reported per lane per call; all warmup and measured "
              "calls matched these exact expected values. Inputs and output traffic are on "
              "separate ports, outside this cache-history/victim accounting.", "",
              "| P | S | Lane | Cache read bytes | Victim write bytes |",
              "| --- | --- | --- | --- | --- |"]
    for position, sequence in CALLS:
        row = next(sample for sample in samples if sample["position"] == position)
        for lane in range(LANES):
            traffic = row["cache_traffic"][str(lane)]
            lines.append(f"| {position} | {sequence} | {lane} | {traffic['read_bytes']} | {traffic['write_bytes']} |")
    lines += ["", "| Numeric comparison | Elements checked | Failed elements | Nonfinite | Max absolute error |",
              "| --- | --- | --- | --- | --- |"]
    for key in ("u50_comparison", "gpu_comparison"):
        checks = [sample[key] for sample in samples]
        lines.append(f"| {key} | {sum(check['elements'] for check in checks)} | "
                     f"{sum(check['failed_elements'] for check in checks)} | "
                     f"{sum(check['nonfinite'] for check in checks)} | "
                     f"{max(check['max_absolute_error'] for check in checks):.8g} |")
    lines += ["", "| HBM writeback check | BF16 words | Bitwise failures |",
              "| --- | --- | --- |"]
    checks = environment["writeback_checks"]
    lines.append(f"| Evicted payloads, every lane and series | {sum(check['bf16_words'] for check in checks)} | "
                 f"{sum(not check['bitwise_passed'] for check in checks)} |")
    lines += ["", "All per-call raw counters, comparisons, transfer/invocation timings, "
              "warmup identities and device outputs are retained in the accompanying evidence files.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--sha256", required=True, help="expected freshly signed-off artifact hash")
    parser.add_argument("--routed-clock-mhz", type=float, required=True)
    parser.add_argument("--routed-clock-report", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True, help="must not already exist")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    require(sys.byteorder == "little", "packed transport requires little-endian host")
    require(args.device >= 0 and args.gpu >= 0 and args.warmup >= 1 and args.repeats >= 3,
            "require valid devices, at least one warmup and several measured series")
    require(0 < args.routed_clock_mhz < 1000 and math.isfinite(args.routed_clock_mhz),
            "invalid routed clock")
    for name in ("model_dir", "xclbin", "routed_clock_report"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    require(re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256) is not None, "invalid expected hash")
    args.sha256 = args.sha256.lower()
    require(sha256(args.xclbin) == args.sha256, "fresh xclbin hash mismatch; no release fallback")
    args.report_dir = args.report_dir.expanduser().resolve()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    output = args.report_dir
    try:
        routed_timing = routed_timing_report(args.routed_clock_report, args.routed_clock_mhz)
        save_json(output / "routed-timing.json", routed_timing)
        clocks = section(args.xclbin, "CLOCK_FREQ_TOPOLOGY", output)
        contract = clock_contract(routed_timing, clocks)
        debug_layout = section(args.xclbin, "DEBUG_IP_LAYOUT", output)
        # Counter-only offload enables monitor counters, without trace-memory
        # traffic. Set before importing pyxrt/XRT; monitor snapshots are deltas.
        ini = output / "xrt.ini"
        ini.write_text("[Debug]\ndevice_counters=true\ndevice_trace=off\n")
        os.environ["XRT_INI_PATH"] = str(ini)
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import numpy as np
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.cuda.set_device(args.gpu)
        gpu = torch.device("cuda", args.gpu)
        require("RTX 5090" in torch.cuda.get_device_name(gpu), "expected RTX 5090")
        with torch.inference_mode():
            qkv = model_inputs(args.model_dir, output, torch, np, gpu)
            references = cpu_reference(qkv, np)
            np.savez(output / "references.npz", **{f"p{p}": words for p, words in references.items()})
            require(sha256(args.xclbin) == args.sha256
                    and sha256(args.routed_clock_report) == routed_timing["report_sha256"],
                    "clock-contract inputs changed before image load")
            require(clock_contract(routed_timing, clocks) == contract,
                    "management clock driver profile changed before image load")
            u50 = U50(args, debug_layout, np, contract)
            rtx = GPU(qkv, torch, gpu)
            environment = {
                "hostname": socket.gethostname(), "started_unix_ns": time.time_ns(),
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(gpu), "loaded_uuid": u50.uuid,
                "pci_path": str(u50.pci), "clock_before": u50.clock_before,
                "routed_clock_report": str(args.routed_clock_report),
                "routed_clock_report_sha256": routed_timing["report_sha256"],
                "routed_timing": routed_timing,
                "clock_contract": contract,
                "xclbin": str(args.xclbin), "xclbin_sha256": args.sha256,
                "script_sha256": sha256(Path(__file__)), "monitor_mapping": u50.monitors.mapping,
                "gpu_initial_input_h2d_host_us": rtx.input_h2d_host_us,
                "timing_note": "AM cycles/routed MHz; CUDA Graph replay of unfused FP32 dense operators; "
                               "internal external-event record nodes exclude host dispatch gaps; "
                               "device-side graph/inter-operator scheduling remains included",
                "numeric_note": "BF16 QKV/O; FP32 QK, standard exp, normalization, SV; TF32 disabled",
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
            save_json(output / "environment.json", environment)
            samples, outputs, writeback_checks = [], {}, []
            with (output / "samples.jsonl").open("w") as journal:
                for series in range(args.warmup + args.repeats):
                    epoch = series + 1
                    for position, sequence in CALLS:
                        gpu_out, gpu_metrics = rtx.call(position, sequence)
                        u50_out, u50_metrics = u50.call(qkv, position, sequence, epoch)
                        sample = {
                            "series": series, "epoch": epoch, "layer": 0,
                            "measured": series >= args.warmup, "position": position, "sequence": sequence,
                            **u50_metrics, **gpu_metrics,
                            "u50_comparison": compare(u50_out, references[position], np),
                            "gpu_comparison": compare(gpu_out, references[position], np),
                        }
                        journal.write(json.dumps(sample, allow_nan=False) + "\n")
                        journal.flush()
                        samples.append(sample)
                        outputs[f"u50_series{series}_p{position}"] = u50_out
                        outputs[f"gpu_series{series}_p{position}"] = gpu_out
                        require(sample["u50_comparison"]["passed"], f"FPGA output mismatch at P={position}")
                        require(sample["gpu_comparison"]["passed"], f"GPU output mismatch at P={position}")
                        require(sample["traffic_passed"], f"cache ownership traffic mismatch at P={position}")
                    checks = u50.verify_writebacks(qkv, series, output)
                    writeback_checks.extend(checks)
                    save_json(output / "writeback-checks.json", writeback_checks)
                    require(all(check["bitwise_passed"] for check in checks), "evicted BF16 writeback payload mismatch")
            environment["clock_after"] = check_clock(u50.pci, contract)
            environment["writeback_checks"] = writeback_checks
            require(u50.device.get_xclbin_uuid().to_string().lower() == u50.uuid.lower(),
                    "device image changed during qualification")
            require(sha256(args.xclbin) == args.sha256, "xclbin file changed during qualification")
            require(sha256(args.routed_clock_report) == routed_timing["report_sha256"],
                    "routed timing report changed during qualification")
            np.savez(output / "outputs.npz", **outputs)
            save_json(output / "environment.json", environment)
            save_json(output / "result.json", {"status": "PASS", "samples": len(samples),
                      "measured_series": args.repeats, "report": "qualification.md"})
            (output / "qualification.md").write_text(markdown_report(args, samples, environment))
        print(f"PASS: {output / 'qualification.md'}")
    except Exception as error:
        if "outputs" in locals():
            np.savez(output / "outputs-before-failure.npz", **outputs)
        if "u50" in locals() and hasattr(u50, "last_snapshot"):
            save_json(output / "last-monitor-snapshot.json", u50.last_snapshot)
        save_json(output / "result.json", {"status": "FAIL", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
