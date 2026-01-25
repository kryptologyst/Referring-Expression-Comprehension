"""
Core utilities for referring expression comprehension.

This module provides essential utilities for device management, seeding,
logging, and configuration handling.
"""

import logging
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        torch.mps.manual_seed(seed)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    OmegaConf.save(config, save_path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configuration objects.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration object
    """
    return OmegaConf.merge(base_config, override_config)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter count and estimated size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": (param_size + buffer_size) / (1024 * 1024),
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset all values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Update with new value.
        
        Args:
            val: New value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]["lr"]
