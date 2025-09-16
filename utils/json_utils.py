"""
This module provides functions to handle NumPy types for JSON serialization.
"""
import numpy as np
from typing import Any, Dict, List, Union

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Any Python object that might contain NumPy types
        
    Returns:
        The same object with NumPy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
