#!/usr/bin/env python3
"""
Demo script for referring expression comprehension.

This script launches the Streamlit demo application.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo."""
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    # Launch Streamlit app
    app_path = src_path / "demo" / "app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
