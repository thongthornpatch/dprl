"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing configuration parameters

    Example:
        >>> config = load_config('configs/config_laptop.yaml')
        >>> print(config['model']['name'])
        'gpt2'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['model', 'data', 'training', 'reward', 'evaluation']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def get_device(config: Dict[str, Any]) -> str:
    """
    Get the device (cpu/cuda) from config.

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cpu' or 'cuda')
    """
    import torch

    device = config.get('hardware', {}).get('device', 'cpu')

    # Fallback to CPU if CUDA requested but not available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    return device


def print_config(config: Dict[str, Any]) -> None:
    """
    Pretty print the configuration.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)

    def _print_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                _print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    _print_dict(config)
    print("=" * 80)


if __name__ == "__main__":
    # Test the config loader
    import sys

    # Try laptop config
    print("Testing laptop config...")
    try:
        config = load_config('configs/config_laptop.yaml')
        print_config(config)
        print("\n✅ Laptop config loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading laptop config: {e}\n")

    # Try NSCC config
    print("Testing NSCC config...")
    try:
        config = load_config('configs/config_nscc.yaml')
        print_config(config)
        print("\n✅ NSCC config loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading NSCC config: {e}\n")
