"""Plotting-Modul: Experiment-Plots und Cross-Experiment-Vergleiche.

Generiert matplotlib-PNG-Plots fuer einzelne Experimente sowie
Vergleichsplots ueber mehrere Experimente hinweg.
"""

from .experiment import generate_experiment_plots
from .comparison import generate_comparison_plots

__all__ = [
    "generate_experiment_plots",
    "generate_comparison_plots",
]
