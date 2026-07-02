"""Device abstraction layer for flexible NPU/CUDA/CPU support.

Auto-detects the available accelerator at import time:
  1. DREAMZERO_DEVICE env var override (npu | cuda | cpu)
  2. torch_npu.is_available() → NPU
  3. torch.cuda.is_available() → CUDA
  4. Fallback to CPU

Provides a unified API so the rest of the codebase can write
device-agnostic code without torch.cuda.* or torch.npu.* calls.
"""

import os
import warnings
from typing import Optional, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# FlashAttention availability (checked once at import time)
# ---------------------------------------------------------------------------
_FLASH_ATTN_2_AVAILABLE = False
_FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn_interface  # noqa: F401
    _FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    pass

try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Internal: detect the preferred accelerator
# ---------------------------------------------------------------------------

def _detect_device_type() -> str:
    """Return 'npu', 'cuda', or 'cpu' based on env and auto-detection."""
    env_device = os.environ.get("DREAMZERO_DEVICE", "").lower()
    if env_device in ("npu", "cuda", "cpu"):
        return env_device

    # Auto-detect: try NPU first, then CUDA, then CPU
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu"
    except (ImportError, AttributeError):
        pass

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ---------------------------------------------------------------------------
# Module-level constants (computed once)
# ---------------------------------------------------------------------------

DEVICE_TYPE: str = _detect_device_type()

_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if DEVICE_TYPE == "npu":
    import torch_npu  # noqa: F401
    DEVICE_STR = f"npu:{_LOCAL_RANK}"
    AUTOCAST_DEVICE = "npu"
    _DIST_BACKEND = "hccl"
    _AcceleratorEvent = torch.npu.Event
    _AcceleratorStream = torch.npu.Stream
    _PROFILER_ACTIVITY = None  # torch_npu profiler activity if available
    try:
        import torch_npu.profiler
        _PROFILER_ACTIVITY = torch_npu.profiler.ProfilerActivity.NPU
    except (ImportError, AttributeError):
        pass

elif DEVICE_TYPE == "cuda":
    DEVICE_STR = f"cuda:{_LOCAL_RANK}"
    AUTOCAST_DEVICE = "cuda"
    _DIST_BACKEND = "nccl"
    _AcceleratorEvent = torch.cuda.Event
    _AcceleratorStream = torch.cuda.Stream
    _PROFILER_ACTIVITY = torch.profiler.ProfilerActivity.CUDA

else:  # cpu
    DEVICE_STR = "cpu"
    AUTOCAST_DEVICE = "cpu"
    _DIST_BACKEND = "gloo"
    _AcceleratorEvent = None
    _AcceleratorStream = None
    _PROFILER_ACTIVITY = None

DEVICE = torch.device(DEVICE_STR)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def is_accelerator_available() -> bool:
    """Return True if any hardware accelerator (NPU or CUDA) is available."""
    if DEVICE_TYPE == "npu":
        try:
            return torch.npu.is_available()
        except (AttributeError, ImportError):
            return False
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.is_available()
    return False


def is_accelerator(device: Union[str, torch.device]) -> bool:
    """Return True if *device* refers to an NPU or CUDA device."""
    if isinstance(device, torch.device):
        return device.type in ("npu", "cuda")
    return str(device) in ("npu", "cuda")


def is_npu() -> bool:
    return DEVICE_TYPE == "npu"


def is_cuda() -> bool:
    return DEVICE_TYPE == "cuda"


def get_device_type() -> str:
    return DEVICE_TYPE


def get_device_capability() -> Tuple[int, int]:
    """Get compute capability.  Returns (0, 0) on NPU/CPU to trigger SDPA fallback."""
    if DEVICE_TYPE == "cuda":
        try:
            return torch.cuda.get_device_capability()
        except Exception:
            return (0, 0)
    return (0, 0)


def get_device_name() -> str:
    """Get human-readable accelerator name."""
    if DEVICE_TYPE == "npu":
        try:
            return torch.npu.get_device_name()
        except Exception:
            return "npu"
    elif DEVICE_TYPE == "cuda":
        try:
            return torch.cuda.get_device_name()
        except Exception:
            return "cuda"
    return "cpu"


def get_device_count() -> int:
    """Return the number of available accelerator devices."""
    if DEVICE_TYPE == "npu":
        try:
            return torch.npu.device_count()
        except Exception:
            return 0
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.device_count()
    return 0


def get_device_properties(device: Optional[Union[str, torch.device]] = None):
    """Return device properties for the accelerator."""
    if DEVICE_TYPE == "npu":
        try:
            return torch.npu.get_device_properties(device or DEVICE)
        except Exception:
            return None
    elif DEVICE_TYPE == "cuda":
        try:
            return torch.cuda.get_device_properties(device or DEVICE)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Synchronization & Events
# ---------------------------------------------------------------------------

if _AcceleratorEvent is not None:
    def Event(*args, **kwargs):
        """Create an accelerator event (torch.npu.Event or torch.cuda.Event)."""
        return _AcceleratorEvent(*args, **kwargs)
else:
    def Event(*args, **kwargs):
        """No-op Event for CPU."""
        return None


def synchronize(device: Optional[Union[str, torch.device]] = None) -> None:
    """Synchronize the accelerator device."""
    if DEVICE_TYPE == "npu":
        torch.npu.synchronize(device)
    elif DEVICE_TYPE == "cuda":
        torch.cuda.synchronize(device)


def current_stream(device: Optional[Union[str, torch.device]] = None):
    """Return the current accelerator stream."""
    if DEVICE_TYPE == "npu":
        return torch.npu.current_stream(device)
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.current_stream(device)
    return None


def set_device(device: Union[str, int, torch.device]) -> None:
    """Set the current accelerator device."""
    if DEVICE_TYPE == "npu":
        torch.npu.set_device(device)
    elif DEVICE_TYPE == "cuda":
        torch.cuda.set_device(device)


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def empty_cache() -> None:
    """Release unoccupied cached memory on the accelerator."""
    if DEVICE_TYPE == "npu":
        torch.npu.empty_cache()
    elif DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()


def mem_get_info(device: Optional[Union[str, torch.device]] = None) -> Tuple[int, int]:
    """Return (free_bytes, total_bytes) for the accelerator device."""
    if DEVICE_TYPE == "npu":
        return torch.npu.mem_get_info(device)
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.mem_get_info(device)
    return (0, 0)


def memory_allocated(device: Optional[Union[str, torch.device]] = None) -> int:
    """Return the current accelerator memory occupied by tensors (bytes)."""
    if DEVICE_TYPE == "npu":
        return torch.npu.memory_allocated(device)
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.memory_allocated(device)
    return 0


def max_memory_allocated(device: Optional[Union[str, torch.device]] = None) -> int:
    """Return the peak accelerator memory occupied by tensors (bytes)."""
    if DEVICE_TYPE == "npu":
        return torch.npu.max_memory_allocated(device)
    elif DEVICE_TYPE == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return 0


def reset_peak_memory_stats(device: Optional[Union[str, torch.device]] = None) -> None:
    """Reset peak memory stats."""
    if DEVICE_TYPE == "npu":
        torch.npu.reset_peak_memory_stats(device)
    elif DEVICE_TYPE == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Memory profiling (NVIDIA-only; no-ops on NPU/CPU)
# ---------------------------------------------------------------------------

_MEMORY_HISTORY_ENABLED = False


def record_memory_history(enabled: bool = True, max_entries: int = 100000) -> None:
    """Enable/disable CUDA memory history recording.  No-op on NPU."""
    global _MEMORY_HISTORY_ENABLED
    if DEVICE_TYPE == "cuda" and enabled:
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all" if enabled else None,
                max_entries=max_entries,
            )
            _MEMORY_HISTORY_ENABLED = True
        except Exception:
            _MEMORY_HISTORY_ENABLED = False
    else:
        _MEMORY_HISTORY_ENABLED = False


def dump_memory_snapshot(path: str) -> None:
    """Dump CUDA memory snapshot.  No-op on NPU."""
    if DEVICE_TYPE == "cuda" and _MEMORY_HISTORY_ENABLED:
        try:
            torch.cuda.memory._dump_snapshot(path)
        except Exception:
            pass
    else:
        warnings.warn("Memory snapshot is only supported on CUDA; skipping.")


# ---------------------------------------------------------------------------
# Seed / determinism
# ---------------------------------------------------------------------------

def manual_seed_all(seed: int) -> None:
    """Set seed on all accelerators."""
    if DEVICE_TYPE == "npu":
        torch.npu.manual_seed_all(seed)
    elif DEVICE_TYPE == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def set_deterministic(flag: bool = True) -> None:
    """Enable deterministic algorithms where possible."""
    if not flag:
        return
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if DEVICE_TYPE == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch.backends.cudnn as cudnn  # noqa: F401
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Distributed / device mesh
# ---------------------------------------------------------------------------

def get_dist_backend() -> str:
    """Return the recommended distributed process group backend."""
    return _DIST_BACKEND


def get_device_mesh_name() -> str:
    """Return the device type string for init_device_mesh."""
    if DEVICE_TYPE == "npu":
        return "npu"
    elif DEVICE_TYPE == "cuda":
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

def get_profiler_activities():
    """Return a list of ProfilerActivity values appropriate for this device."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if _PROFILER_ACTIVITY is not None:
        activities.append(_PROFILER_ACTIVITY)
    return activities


# ---------------------------------------------------------------------------
# Patch torch.distributed.init_process_group so that libraries (HF Trainer,
# Accelerate, etc.) that hardcode backend="nccl" automatically use HCCL
# on NPU.  This is a one-shot monkey-patch applied at import time.
# ---------------------------------------------------------------------------

_original_init_process_group = torch.distributed.init_process_group


def _patched_init_process_group(*args, **kwargs):
    """Auto-correct backend='nccl' → 'hccl' on NPU devices."""
    if DEVICE_TYPE == "npu":
        if kwargs.get("backend", "") == "nccl":
            kwargs["backend"] = "hccl"
        elif len(args) >= 1 and args[0] == "nccl":
            args = ("hccl",) + args[1:]
    return _original_init_process_group(*args, **kwargs)


if DEVICE_TYPE == "npu":
    torch.distributed.init_process_group = _patched_init_process_group


# ---------------------------------------------------------------------------
# FlashAttention compatibility
# ---------------------------------------------------------------------------

def gpu_supports_flash_attention() -> bool:
    """Check if the current accelerator supports FlashAttention.

    On NPU: always returns False so SDPA fallback is used.
      torch_npu natively supports F.scaled_dot_product_attention,
      so no custom kernel is needed.

    On CUDA: requires Ampere (compute capability >= 8.0) and
      flash_attn 2 or flash_attn 3 to be installed.
    """
    if DEVICE_TYPE != "cuda":
        return False
    if not (_FLASH_ATTN_2_AVAILABLE or _FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False
