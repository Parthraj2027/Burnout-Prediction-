"""Compatibility imports for older scripts.

The training code lives in ``src``.  Keeping these names here means older
commands that import ``model`` continue to work without maintaining a second
implementation.
"""

from src.data_processor import DataProcessor as BurnoutDatasetMerger
from src.single_model_trainer import BurnoutPredictor
from src.main import main

__all__ = ["BurnoutDatasetMerger", "BurnoutPredictor", "main"]


if __name__ == "__main__":
    main()
