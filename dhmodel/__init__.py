"""
DH-model: Deleuze et Houllier 1998 model with modifications (Python version)

A process-based xylem growth model for simulating wood density and ring increment.
"""

from .run_dh_model import run_dh_model
from .daylength import daylength
from .data import load_dh_model_ins

__version__ = "1.0.0"
__author__ = "Annemarie Eckes-Shephard"
__all__ = ["run_dh_model", "daylength", "load_dh_model_ins"]
