"""Backend runtime helpers for AMMC Gen-5.

The project started CUDA-first because Colab T4 was the easiest accelerator to
reach. This module makes the runtime accelerator-neutral and gives TPU/XLA a
first-class path without forcing every model file to import ``torch_xla``
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import SimpleNamespace
from typing import Any

try:  # pragma: no cover - exercised in accelerator runtimes
    import torch
except Exception:  # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 runtime helpers require PyTorch")


@dataclass(frozen=True)
class AcceleratorMemory:
    """Best-effort accelerator memory reading.

    CUDA exposes direct allocation counters. PyTorch/XLA does not expose a
    stable Colab TPU memory API with the same semantics, so XLA currently
    returns ``None`` fields rather than pretending the numbers are comparable.
    """

    allocated_mb: float | None
    max_allocated_mb: float | None
    backend: str


def resolve_device(device: str | Any = "auto"):
    """Resolve ``cpu``, ``cuda``, or ``xla`` into a PyTorch device object.

    ``auto`` intentionally prefers XLA when PyTorch/XLA can acquire a device,
    then CUDA, then CPU. This matches the Gen-5 TPU-first roadmap while keeping
    Colab T4 and local CPU runs available as fallbacks.
    """

    _require_torch()
    if not isinstance(device, str):
        return device

    requested = device.lower()
    if requested in {"xla", "tpu"}:
        return xla_device()
    if requested == "auto":
        xla = try_xla_device()
        if xla is not None:
            return xla
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def try_xla_device():
    """Return an XLA device if PyTorch/XLA is importable and initialized."""

    try:
        return xla_device()
    except Exception:
        return None


def xla_device():
    """Return the current XLA device using modern or legacy PyTorch/XLA APIs."""

    try:
        torch_xla = importlib.import_module("torch_xla")
    except Exception as exc:
        raise ImportError(_xla_dependency_message(exc)) from exc
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
    except Exception as exc:
        raise ImportError(_xla_dependency_message(exc)) from exc
    return xm.xla_device()


def _xla_dependency_message(original_error: BaseException | None = None) -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    torch_version = getattr(torch, "__version__", "unknown") if torch is not None else "unavailable"
    original = f" Original import error: {original_error}" if original_error is not None else ""
    return (
        "PyTorch/XLA is required for --device xla/--device tpu, but the "
        f"`torch_xla` package could not be loaded in this Python {version}, "
        f"PyTorch {torch_version} runtime. This usually means either "
        "`torch_xla` is missing or its binary was built for a different "
        "PyTorch/Python ABI. In Colab, switch to a TPU runtime and install a "
        "PyTorch/XLA build that matches the active Python/PyTorch version, "
        "then restart the runtime before rerunning. If this Colab instance "
        "only has a T4/L4 GPU, rerun with `--device cuda` instead of "
        f"`--device xla`.{original}"
    )


def is_xla_device(device) -> bool:
    if device is None:
        return False
    return getattr(device, "type", None) == "xla" or str(device).startswith("xla")


def is_cuda_device(device) -> bool:
    return getattr(device, "type", None) == "cuda"


def device_kind(device) -> str:
    if is_xla_device(device):
        return "xla"
    if is_cuda_device(device):
        return "cuda"
    return getattr(device, "type", str(device))


def seed_everything(seed: int, device=None) -> None:
    """Seed global RNGs for CPU/CUDA/XLA best-effort reproducibility."""

    _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
    if is_xla_device(device):
        try:
            xm = importlib.import_module("torch_xla.core.xla_model")
            if hasattr(xm, "set_rng_state"):
                xm.set_rng_state(int(seed), device=device)
        except Exception:
            pass


def make_generator(seed: int, device=None):
    """Create a per-device generator when the backend supports one.

    PyTorch/XLA random operations generally use global XLA RNG state rather than
    ``torch.Generator(device='xla')``. Returning ``None`` for XLA keeps tensor
    random calls on the accelerator and relies on ``seed_everything``.
    """

    _require_torch()
    if is_xla_device(device):
        seed_everything(seed, device=device)
        return None
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def mark_step(device=None) -> None:
    """Tell lazy accelerators that a logical step is complete."""

    if not is_xla_device(device):
        return
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        if hasattr(xm, "mark_step"):
            xm.mark_step()
            return
    except Exception:
        pass
    try:
        torch_xla = importlib.import_module("torch_xla")
        step = getattr(torch_xla, "step", None)
        if callable(step):
            step()
    except Exception:
        pass


def sync(device=None) -> None:
    """Synchronize accelerator work for benchmark timing and output reads."""

    _require_torch()
    if is_cuda_device(device):
        torch.cuda.synchronize(device)
        return
    if not is_xla_device(device):
        return
    try:
        torch_xla = importlib.import_module("torch_xla")
        sync_fn = getattr(torch_xla, "sync", None)
        if callable(sync_fn):
            sync_fn()
            return
    except Exception:
        pass
    try:
        xm = importlib.import_module("torch_xla.core.xla_model")
        if hasattr(xm, "mark_step"):
            xm.mark_step(wait=True)
    except Exception:
        mark_step(device)


def clear_memory_stats(device=None) -> None:
    """Reset accelerator memory counters when the backend supports it."""

    _require_torch()
    if is_cuda_device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def accelerator_memory(device=None) -> AcceleratorMemory:
    """Return comparable CUDA memory stats or explicit nulls for other backends."""

    _require_torch()
    if is_cuda_device(device):
        return AcceleratorMemory(
            allocated_mb=torch.cuda.memory_allocated(device) / (1024 * 1024),
            max_allocated_mb=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            backend="cuda",
        )
    return AcceleratorMemory(allocated_mb=None, max_allocated_mb=None, backend=device_kind(device))


def memory_namespace(device=None):
    """Backward-compatible namespace used by older benchmark code."""

    memory = accelerator_memory(device)
    return SimpleNamespace(
        allocated_mb=memory.allocated_mb,
        max_allocated_mb=memory.max_allocated_mb,
        backend=memory.backend,
    )
