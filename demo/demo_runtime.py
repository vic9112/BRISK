#********************************************************
#* Project         : FPGenius - FPT'26 Design Competition Track B
#* Filename        : demo_runtime.py
#* Author          : VicChen (KUAN-HSI, CHEN)
#* Created On      : 2026/07/10
#* Function        : Run the standalone live RTX 5090 and U50 Attention comparison.
#********************************************************

"""Standalone runtime for the FPT'26 attention-only live demonstration.

The module deliberately imports accelerator libraries only during hardware
setup.  Configuration and packaging tests therefore run on a machine without
CUDA, XRT, or the model checkpoint.
"""

from __future__ import annotations

import atexit
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple


MODEL_ENV = "FPT26_MODEL_DIR"
XCLBIN_ENV = "FPT26_XCLBIN"
DEVICE_ENV = "FPT26_DEVICE"
KERNEL_NAME = "llama3_fast_attention"
MAX_SEQUENCE = 128
MAX_ACTIVE_SEQUENCE = 8
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIMENSION = 128
LANES = 4
PAIRS_PER_HEAD = 16
METRICS_NOTE = (
    "The displayed ratio divides comparable host-observed invocation envelopes: "
    "RTX Attention call through CUDA completion versus U50 kernel submission "
    "through XRT wait. RTX CUDA-event diagnostic time is reported separately "
    "and is never used in the ratio."
)
_DISTRIBUTED_CLEANUP_REGISTERED = False


def _destroy_meta_process_group() -> None:
    torch = importlib.import_module("torch")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def comparison_metrics(
    *,
    rtx_host_invocation_us: float,
    u50_host_invocation_us: float,
    rtx_cuda_event_us: float,
) -> dict[str, float]:
    """Build a comparison without mixing host-envelope and device-event scopes."""

    values = (
        float(rtx_host_invocation_us),
        float(u50_host_invocation_us),
        float(rtx_cuda_event_us),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("live Attention timings must be finite and positive")
    return {
        "pure_rtx_host_invocation_us": values[0],
        "hybrid_u50_host_invocation_us": values[1],
        "host_invocation_ratio": values[0] / values[1],
        "pure_rtx_cuda_event_us": values[2],
    }


class DemoConfigurationError(RuntimeError):
    """Raised before a benchmark when required local inputs are invalid."""


class DemoConfig(NamedTuple):
    model_dir: Path
    xclbin: Path
    device_index: int
    model_format: str
    official_package_dir: Path | None


def _required_path(variable: str, *, directory: bool) -> Path:
    raw = os.environ.get(variable, "").strip()
    if not raw:
        raise DemoConfigurationError(f"{variable} is required")
    path = Path(raw).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    expected = "directory" if directory else "file"
    if not valid:
        raise DemoConfigurationError(f"{variable} must name an existing {expected}: {path}")
    return path


def load_config() -> DemoConfig:
    """Read the three public inputs without importing hardware libraries."""

    model_dir = _required_path(MODEL_ENV, directory=True)
    official_package_dir: Path | None = None
    if (model_dir / "config.json").is_file():
        model_format = "huggingface"
    elif all(
        (model_dir / name).is_file()
        for name in ("params.json", "consolidated.00.pth", "tokenizer.model")
    ):
        model_format = "meta"
        official_package_dir = (model_dir.parent / "llama").resolve()
        if not all(
            (official_package_dir / name).is_file()
            for name in ("__init__.py", "generation.py", "model.py", "tokenizer.py")
        ):
            raise DemoConfigurationError(
                f"official Meta checkpoint requires the sibling llama source package: {official_package_dir}"
            )
    else:
        raise DemoConfigurationError(
            f"{MODEL_ENV} must contain either config.json or the official Meta "
            "params.json, consolidated.00.pth, and tokenizer.model files"
        )
    xclbin = _required_path(XCLBIN_ENV, directory=False)
    raw_device = os.environ.get(DEVICE_ENV, "0").strip()
    try:
        device_index = int(raw_device)
    except ValueError as exc:
        raise DemoConfigurationError(f"{DEVICE_ENV} must be a non-negative integer") from exc
    if device_index < 0:
        raise DemoConfigurationError(f"{DEVICE_ENV} must be a non-negative integer")
    return DemoConfig(
        model_dir=model_dir,
        xclbin=xclbin,
        device_index=device_index,
        model_format=model_format,
        official_package_dir=official_package_dir,
    )


def preflight(config: DemoConfig, *, check_hardware: bool = True) -> dict[str, str]:
    """Validate files, then optionally validate the exact GPU/model/XRT stack."""

    if not config.model_dir.is_dir():
        raise DemoConfigurationError("model directory is no longer readable")
    if not config.xclbin.is_file():
        raise DemoConfigurationError("xclbin is no longer readable")
    report = {
        "model_directory": "ready",
        "model_format": config.model_format,
        "xclbin": "ready",
        "device_checks": "skipped",
    }
    if not check_hardware:
        return report

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise DemoConfigurationError("PyTorch is not installed; install requirements.md") from exc
    if not torch.cuda.is_available():
        raise DemoConfigurationError("CUDA is unavailable; Pure RTX 5090 cannot run")
    gpu_name = str(torch.cuda.get_device_name(0))
    if "RTX 5090" not in gpu_name:
        raise DemoConfigurationError(f"expected an RTX 5090, detected {gpu_name}")

    try:
        importlib.import_module("pyxrt")
    except ImportError as exc:
        raise DemoConfigurationError(
            "pyxrt is unavailable; install AMD XRT and activate its Python environment"
        ) from exc

    try:
        if config.model_format == "huggingface":
            transformers = importlib.import_module("transformers")
            model_config = transformers.AutoConfig.from_pretrained(
                config.model_dir, local_files_only=True
            )
            transformers.AutoTokenizer.from_pretrained(
                config.model_dir, local_files_only=True
            )
            hidden_size = int(getattr(model_config, "hidden_size", 0))
            query_heads = int(getattr(model_config, "num_attention_heads", 0))
            kv_heads = int(getattr(model_config, "num_key_value_heads", 0))
        else:
            params = json.loads(
                (config.model_dir / "params.json").read_text(encoding="utf-8")
            )
            importlib.import_module("fairscale")
            importlib.import_module("tiktoken")
            hidden_size = int(params.get("dim", 0))
            query_heads = int(params.get("n_heads", 0))
            kv_heads = int(params.get("n_kv_heads", query_heads))
    except Exception as exc:
        raise DemoConfigurationError(
            f"the local {config.model_format} model package is incomplete or unreadable"
        ) from exc

    head_dimension = hidden_size // query_heads if query_heads else 0
    observed = (query_heads, kv_heads, head_dimension)
    expected = (NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIMENSION)
    if observed != expected:
        raise DemoConfigurationError(
            f"model attention geometry must be {expected}, observed {observed}"
        )

    report["device_checks"] = "ready"
    report["gpu"] = gpu_name
    report["u50_device"] = str(config.device_index)
    return report


class _U50Adapter:
    """Minimal direct XRT binding for the public 22-argument attention port."""

    def __init__(self, config: DemoConfig) -> None:
        self._np = importlib.import_module("numpy")
        self._xrt = importlib.import_module("pyxrt")
        self._device = self._xrt.device(config.device_index)
        self._uuid = self._device.load_xclbin(str(config.xclbin))
        self._kernel = self._xrt.kernel(
            self._device, self._uuid, KERNEL_NAME
        )
        cache_bytes = MAX_SEQUENCE * 2 * PAIRS_PER_HEAD * 32
        self._cache_buffers = [self._buffer(cache_bytes, 5 + lane) for lane in range(LANES)]
        self._cache_identity = 0

    def _buffer(self, size: int, argument: int) -> Any:
        # Device-backed BOs work on a stock U50 configuration.  host_only BOs
        # require the optional XRT host-memory bank to be enabled system-wide.
        flags = self._xrt.bo.flags.normal
        return self._xrt.bo(
            self._device, size, flags, self._kernel.group_id(argument)
        )

    def _to_device(self, buffer: Any, payload: bytes) -> None:
        buffer.write(payload, 0)
        direction = self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        buffer.sync(direction, len(payload), 0)

    def _from_device(self, buffer: Any, size: int) -> bytes:
        direction = self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        buffer.sync(direction, size, 0)
        return bytes(buffer.read(size, 0))

    def attention(self, q: Any, k: Any, v: Any) -> tuple[Any, float]:
        np = self._np
        batch, sequence, query_heads, head_dimension = q.shape
        if (batch, query_heads, head_dimension) != (1, NUM_QUERY_HEADS, HEAD_DIMENSION):
            raise DemoConfigurationError("the public U50 runtime requires [1, S, 32, 128] Q")
        if tuple(k.shape) != (1, sequence, NUM_KV_HEADS, HEAD_DIMENSION):
            raise DemoConfigurationError("the public U50 runtime requires [1, S, 8, 128] K/V")
        if sequence < 1 or sequence > MAX_ACTIVE_SEQUENCE:
            raise DemoConfigurationError("active prompt length must be between 1 and 8 tokens")

        # A BF16 tensor can be viewed as unsigned 16-bit storage without a
        # numeric conversion.  NumPy then supplies the exact bytes to XRT.
        torch = importlib.import_module("torch")
        q_words = q.contiguous().view(torch.uint16).numpy()
        k_words = k.contiguous().view(torch.uint16).numpy()
        v_words = v.contiguous().view(torch.uint16).numpy()

        query_buffers: list[Any] = []
        output_buffers: list[Any] = []
        heads_per_lane = NUM_QUERY_HEADS // LANES
        lane_bytes = batch * sequence * heads_per_lane * HEAD_DIMENSION * 2
        for lane in range(LANES):
            start = lane * heads_per_lane
            stop = start + heads_per_lane
            payload = np.ascontiguousarray(q_words[:, :, start:stop, :]).tobytes()
            query_buffer = self._buffer(lane_bytes, lane)
            self._to_device(query_buffer, payload)
            query_buffers.append(query_buffer)
            output_buffers.append(self._buffer(lane_bytes, 9 + lane))

        paired = np.empty(
            (batch, sequence, NUM_KV_HEADS, PAIRS_PER_HEAD, 16), dtype=np.uint16
        )
        reshaped_k = k_words.reshape(
            batch, sequence, NUM_KV_HEADS, PAIRS_PER_HEAD, 8
        )
        reshaped_v = v_words.reshape(
            batch, sequence, NUM_KV_HEADS, PAIRS_PER_HEAD, 8
        )
        paired[..., :8] = reshaped_k
        paired[..., 8:] = reshaped_v
        active_payload = paired.tobytes()
        active_buffer = self._buffer(len(active_payload), 4)
        self._to_device(active_buffer, active_payload)

        status_buffer = self._buffer(4, 21)
        arguments = [
            *query_buffers,
            active_buffer,
            *self._cache_buffers,
            *output_buffers,
            batch,
            sequence,
            0,
            MAX_SEQUENCE,
            0,
            1,
            0,
            self._cache_identity,
            status_buffer,
        ]
        if len(arguments) != 22:
            raise AssertionError("U50 argument table must contain 22 entries")
        self._cache_identity += 1 << 5  # Fresh layer-zero prompt, new epoch.

        started = time.perf_counter_ns()
        run = self._kernel(*arguments)
        # Request a wait timeout, not a hard wall-clock bound or hardware abort.
        state = run.wait(10_000)
        host_invocation_us = (time.perf_counter_ns() - started) / 1_000.0
        if state != self._xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(
                f"U50 XRT run did not complete: {state}; timeout does not abort; no automatic abort/reset"
            )
        status_code = int.from_bytes(
            self._from_device(status_buffer, 4), byteorder="little", signed=True
        )
        if status_code != 0:
            raise RuntimeError(f"U50 kernel rejected the live request with status {status_code}")

        output = np.empty((batch, sequence, NUM_QUERY_HEADS, HEAD_DIMENSION), dtype=np.uint16)
        for lane, buffer in enumerate(output_buffers):
            lane_words = np.frombuffer(self._from_device(buffer, lane_bytes), dtype=np.uint16)
            lane_words = lane_words.reshape(batch, sequence, heads_per_lane, HEAD_DIMENSION)
            start = lane * heads_per_lane
            output[:, :, start : start + heads_per_lane, :] = lane_words
        return torch.from_numpy(output).view(torch.bfloat16), host_invocation_us


def _attention_reference(torch: Any, q: Any, k: Any, v: Any) -> Any:
    repeats = NUM_QUERY_HEADS // NUM_KV_HEADS
    expanded_k = k.repeat_interleave(repeats, dim=2)
    expanded_v = v.repeat_interleave(repeats, dim=2)
    query = q.transpose(1, 2).float()
    key = expanded_k.transpose(1, 2).float()
    value = expanded_v.transpose(1, 2).float()
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(HEAD_DIMENSION)
    sequence = q.shape[1]
    causal = torch.ones((sequence, sequence), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    # Only the interface is BF16: scores, probabilities and SV remain FP32.
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value).to(q.dtype).transpose(1, 2).contiguous()


class DemoSession:
    """Loaded strict hardware session used by the notebook button callback."""

    def __init__(self, config: DemoConfig, report: dict[str, str]) -> None:
        self.config = config
        self.report = report
        self._torch = importlib.import_module("torch")
        self._torch.backends.cuda.matmul.allow_tf32 = False
        self._torch.backends.cudnn.allow_tf32 = False
        self._torch.set_float32_matmul_precision("highest")
        if config.model_format == "huggingface":
            self._initialize_huggingface()
        else:
            self._initialize_meta()
        self._u50 = _U50Adapter(config)

    def _initialize_huggingface(self) -> None:
        transformers = importlib.import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_dir, local_files_only=True
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            local_files_only=True,
            torch_dtype=self._torch.bfloat16,
        ).to("cuda")
        self._model.eval()
        llama_module = importlib.import_module(
            "transformers.models.llama.modeling_llama"
        )
        self._apply_rotary_pos_emb = llama_module.apply_rotary_pos_emb

    def _initialize_meta(self) -> None:
        global _DISTRIBUTED_CLEANUP_REGISTERED

        package_dir = self.config.official_package_dir
        if package_dir is None:
            raise DemoConfigurationError("official Meta llama package was not resolved")
        package_root = str(package_dir.parent)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29537")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        llama = importlib.import_module("llama")
        loaded_package = Path(llama.__file__).resolve().parent
        if loaded_package != package_dir:
            raise DemoConfigurationError(
                f"resolved llama package {loaded_package}, expected {package_dir}"
            )
        generator = llama.Llama.build(
            ckpt_dir=str(self.config.model_dir),
            tokenizer_path=str(self.config.model_dir / "tokenizer.model"),
            max_seq_len=MAX_ACTIVE_SEQUENCE,
            max_batch_size=1,
            model_parallel_size=1,
            seed=1,
        )
        if not _DISTRIBUTED_CLEANUP_REGISTERED:
            atexit.register(_destroy_meta_process_group)
            _DISTRIBUTED_CLEANUP_REGISTERED = True
        self._model = generator.model
        self._tokenizer = generator.tokenizer
        meta_model = importlib.import_module("llama.model")
        self._apply_rotary_emb = meta_model.apply_rotary_emb

    def _prompt_qkv_huggingface(
        self, prompt: str
    ) -> tuple[list[int], Any, Any, Any]:
        torch = self._torch
        encoded = self._tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_ACTIVE_SEQUENCE,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to("cuda")
        if input_ids.shape[1] < 1:
            raise ValueError("the tokenizer produced no prompt tokens")

        with torch.inference_mode():
            backbone = self._model.model
            layer = backbone.layers[0]
            attention = layer.self_attn
            hidden_states = backbone.embed_tokens(input_ids)
            hidden_states = layer.input_layernorm(hidden_states)
            batch, sequence, _ = hidden_states.shape
            query = attention.q_proj(hidden_states).view(
                batch, sequence, NUM_QUERY_HEADS, HEAD_DIMENSION
            ).transpose(1, 2)
            key = attention.k_proj(hidden_states).view(
                batch, sequence, NUM_KV_HEADS, HEAD_DIMENSION
            ).transpose(1, 2)
            value = attention.v_proj(hidden_states).view(
                batch, sequence, NUM_KV_HEADS, HEAD_DIMENSION
            ).transpose(1, 2)
            position_ids = torch.arange(sequence, device="cuda").unsqueeze(0)
            cos, sin = backbone.rotary_emb(hidden_states, position_ids=position_ids)
            query, key = self._apply_rotary_pos_emb(query, key, cos, sin)
        return (
            [int(token) for token in input_ids[0].tolist()],
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
        )

    def _prompt_qkv_meta(self, prompt: str) -> tuple[list[int], Any, Any, Any]:
        torch = self._torch
        token_ids = self._tokenizer.encode(prompt, bos=True, eos=False)
        token_ids = token_ids[:MAX_ACTIVE_SEQUENCE]
        if not token_ids:
            raise ValueError("the tokenizer produced no prompt tokens")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden_states = self._model.tok_embeddings(input_ids)
            layer = self._model.layers[0]
            hidden_states = layer.attention_norm(hidden_states)
            attention = layer.attention
            batch, sequence, _ = hidden_states.shape
            query = attention.wq(hidden_states).view(
                batch, sequence, NUM_QUERY_HEADS, HEAD_DIMENSION
            )
            key = attention.wk(hidden_states).view(
                batch, sequence, NUM_KV_HEADS, HEAD_DIMENSION
            )
            value = attention.wv(hidden_states).view(
                batch, sequence, NUM_KV_HEADS, HEAD_DIMENSION
            )
            frequencies = self._model.freqs_cis.to(hidden_states.device)[:sequence]
            query, key = self._apply_rotary_emb(query, key, frequencies)
        return token_ids, query.contiguous(), key.contiguous(), value.contiguous()

    def _prompt_qkv(self, prompt: str) -> tuple[list[int], Any, Any, Any]:
        if self.config.model_format == "huggingface":
            return self._prompt_qkv_huggingface(prompt)
        return self._prompt_qkv_meta(prompt)

    def benchmark(self, prompt: str) -> dict[str, Any]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt cannot be empty")
        torch = self._torch
        token_ids, q_gpu, k_gpu, v_gpu = self._prompt_qkv(clean_prompt)
        for _ in range(2):
            _attention_reference(torch, q_gpu, k_gpu, v_gpu)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_started = time.perf_counter_ns()
        start.record()
        pure_output = _attention_reference(torch, q_gpu, k_gpu, v_gpu)
        end.record()
        torch.cuda.synchronize()
        pure_host_invocation_us = (time.perf_counter_ns() - host_started) / 1_000.0
        pure_cuda_event_us = float(start.elapsed_time(end) * 1_000.0)

        # The hybrid boundary receives the tensors produced on the GPU through
        # CPU/XRT staging, then executes Attention on U50.
        hybrid_output, hybrid_host_invocation_us = self._u50.attention(
            q_gpu.cpu(), k_gpu.cpu(), v_gpu.cpu()
        )
        reference = pure_output.cpu().float()
        observed = hybrid_output.float()
        error = (reference - observed).abs()
        max_error = float(error.max().item())
        finite = torch.isfinite(reference) & torch.isfinite(observed)
        relative_error = error / reference.abs().clamp_min(1e-6)
        if not bool((finite & ((error <= 0.0625) | (relative_error <= 0.08))).all().item()):
            raise RuntimeError(
                f"live U50 output check failed: maximum absolute error {max_error:.6f}"
            )
        metrics = comparison_metrics(
            rtx_host_invocation_us=pure_host_invocation_us,
            u50_host_invocation_us=hybrid_host_invocation_us,
            rtx_cuda_event_us=pure_cuda_event_us,
        )
        return {
            "prompt_tokens": len(token_ids),
            **metrics,
            "max_absolute_error": max_error,
            "correctness_passed": True,
            "measurement_note": METRICS_NOTE,
        }


def initialize_demo() -> DemoSession:
    """Perform strict setup and return the live session; errors stop the demo."""

    config = load_config()
    report = preflight(config, check_hardware=True)
    return DemoSession(config, report)
