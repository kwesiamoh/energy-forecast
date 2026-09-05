"""Small shared helpers for foundation-model device and dtype selection."""

from dataclasses import dataclass


CUDA_UNAVAILABLE = (
    "CUDA was requested but torch.cuda.is_available() is False. "
    "Use device='cpu' or install a CUDA-enabled PyTorch build."
)


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved PyTorch execution device."""

    name: str
    gpu_name: str | None = None


def resolve_device(device: str = "auto") -> DeviceSpec:
    """Resolve ``auto``/``cpu``/``cuda`` without silently changing requests."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Foundation-model inference requires PyTorch. Install a PyTorch "
            "build appropriate for this machine."
        ) from exc

    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(CUDA_UNAVAILABLE)

    resolved = "cuda" if requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    ) else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if resolved == "cuda" else None
    return DeviceSpec(name=resolved, gpu_name=gpu_name)


def resolve_dtype(
    dtype: str,
    device: DeviceSpec,
    *,
    reduced_precision: bool = True,
):
    """Return a safe dtype name and ``torch.dtype`` for the selected device."""
    import torch

    requested = dtype.lower()
    if requested not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(
            "dtype must be one of: 'auto', 'float32', 'float16', 'bfloat16'."
        )

    if not reduced_precision:
        if requested not in {"auto", "float32"}:
            raise ValueError(
                "This model's supported inference path uses float32; "
                "choose dtype='auto' or dtype='float32'."
            )
        return "float32", torch.float32

    if device.name == "cpu":
        if requested not in {"auto", "float32"}:
            raise ValueError(
                "CPU inference uses float32 for portability; choose "
                "dtype='auto' or dtype='float32'."
            )
        return "float32", torch.float32

    if requested == "auto":
        requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if requested == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "bfloat16 was requested, but the selected CUDA device does not "
            "report bfloat16 support. Use dtype='float16' or 'float32'."
        )

    return requested, {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
