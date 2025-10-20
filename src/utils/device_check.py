import torch


def get_device():
    """
    Returns the device to use:
    - 'mps' for Mac M1 GPU if available
    - 'cuda' for NVIDIA GPU
    - 'cpu' as fallback
    """
    if torch.backends.mps.is_available():
        return "cpu"  # Use CPU for stability to avoid SIGBUS on M1
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
