#!/usr/bin/env python3
"""
Setup script for referring expression comprehension project.

This script helps set up the project environment and dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Referring Expression Comprehension project...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data/images",
        "outputs",
        "logs",
        "checkpoints",
        "assets",
        "tests",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        print("Failed to install basic requirements")
        sys.exit(1)
    
    # Install project in development mode
    if not run_command("pip install -e .", "Installing project in development mode"):
        print("Failed to install project")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("Warning: Failed to install development dependencies")
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    if not run_command("python train.py --create_synthetic", "Creating synthetic dataset"):
        print("Warning: Failed to create synthetic dataset")
    
    # Run tests
    print("\nRunning tests...")
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Warning: Some tests failed")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python demo.py")
    print("2. Train a model: python train.py --epochs 10")
    print("3. Evaluate a model: python evaluate.py --checkpoint outputs/best_model.pth")
    print("4. Run benchmark: python benchmark.py --epochs 10")


if __name__ == "__main__":
    main()
