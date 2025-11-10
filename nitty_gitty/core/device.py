import numpy as np
from enum import Enum

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"

DTYPE_MAP = {
    # Floats
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    float: np.float64,  # Default Python float is 64-bit
    
    # Integers
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    int: np.int64,      # Default Python int is 64-bit

    # Unsigned Integers
    'uint8': np.uint8,

    # Booleans
    'bool': np.bool_,
    bool: np.bool_
}

def normalize_dtype(dtype=None) -> np.dtype:
    if dtype is None:
        # Default to float32, a common standard in deep learning
        return np.float32

    # If it's already a numpy dtype, return it
    if isinstance(dtype, np.dtype):
        return dtype

    # Look up the dtype in our map
    normalized = DTYPE_MAP.get(dtype)
    if normalized:
        return normalized
    
    # If not in the map, try to let numpy handle it directly (e.g., np.float32)
    try:
        return np.dtype(dtype)
    except TypeError:
        raise TypeError(f"Invalid or unsupported dtype specified: {dtype}")
    

ENGINE_MAP = {
    Device.CPU: np
}
if CUPY_AVAILABLE:
    ENGINE_MAP[Device.GPU] = cp


def normalize_device(device: str | Device) -> Device:
    if isinstance(device, Device):
        if device == Device.GPU and not is_cuda_available():
            raise RuntimeError("Cannot use GPU device: CuPy is not available.")
        return device
    
    if isinstance(device, str):
        device_lower = device.lower()
        if device_lower == 'cpu':
            return Device.CPU
        if device_lower in ('cuda', 'gpu'):
            if not is_cuda_available():
                raise RuntimeError("Cannot use GPU/CUDA device: CuPy is not available.")
            return Device.GPU
    
    raise ValueError(f"Invalid device specified: {device}")


def get_numpy_engine():
    return np

def get_cupy_engine():
    return cp

def is_cuda_available():
    return CUPY_AVAILABLE

def get_array_module(x):
    return cp if CUPY_AVAILABLE and isinstance(x, cp.ndarray) else np


def get_device_from_array(x) -> Device:
    if CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return Device.GPU
    return Device.CPU


def check_device_compatibility(*arrays):
    if not arrays:
        return Device.CPU # Default to CPU if no arrays are provided
    
    devices = {get_device_from_array(arr) for arr in arrays}
    if len(devices) > 1:
        raise ValueError(f"Device mismatch among inputs: {devices}")
    return devices.pop()