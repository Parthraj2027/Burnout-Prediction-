import sys
import os
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

cache_dir = Path(__file__).parent / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache_dir)

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.main import main

if __name__ == "__main__":
    print("Starting Burnout Prediction Pipeline...\n")
    output = main()
    print("\nPipeline execution completed successfully!")
