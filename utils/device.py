"""
Platform and GPU detection for embedding model placement.
Detects Apple Silicon (MPS), NVIDIA (CUDA), or falls back to CPU.
Result is cached after first call.
"""
import platform
import click

_detected_device = None

VALID_DEVICES = ('cpu', 'mps', 'cuda')

DEVICE_LABELS = {
    'mps': 'Apple Silicon GPU (MPS)',
    'cuda': 'NVIDIA GPU (CUDA)',
    'cpu': 'CPU',
}


def detect_device(config_override=None):
    """
    Detect the best available compute device for PyTorch.

    Args:
        config_override: Optional string from config.yaml ('cpu', 'mps', 'cuda').
                         If provided and valid, skips auto-detection.

    Returns:
        A string: 'mps', 'cuda', or 'cpu'
    """
    global _detected_device
    if _detected_device is not None:
        return _detected_device

    # Honor manual override from config
    if config_override and config_override in VALID_DEVICES:
        _detected_device = config_override
        click.echo(f"INFO: Device override from config: '{config_override}'", err=True)
        return _detected_device

    # Auto-detect
    device = 'cpu'
    try:
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
    except ImportError:
        click.echo("WARNING: torch not installed; defaulting to CPU.", err=True)
    except Exception as e:
        click.echo(f"WARNING: Device detection failed ({e}); defaulting to CPU.", err=True)

    _detected_device = device
    click.echo(f"INFO: Compute device: {DEVICE_LABELS.get(device, device)}", err=True)

    return _detected_device


def get_device_info():
    """
    Returns a dict with platform details for diagnostics.
    """
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'device': _detected_device or 'not yet detected',
    }
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['mps_available'] = torch.backends.mps.is_available()
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except (ImportError, OSError):
        info['torch_version'] = 'not installed'
        info['mps_available'] = False
        info['cuda_available'] = False

    return info
