"""kmz: prediction and reduction of nonlinear dynamical systems from data, with the Koopman
operator and the Mori–Zwanzig formalism.

Minimal usage:

    from kmz import Data, KMZ, find_basins, analyze
    data = Data(trajectories, dt=0.1, observable=0)   # trajectories: (N, T+1, d)
    model = KMZ().fit(data)
    pred = model.predict(x0)          # pred.value, pred.error, pred.t
    model.evaluate_dictionary()       # judge the dictionary, get advice
    basins = find_basins(data)        # basins.accuracy, basins.plot("map.png")
    report = analyze(data)            # understand: invariants, Koopman modes, memory, predictability
"""
from .data import Data
from .model import KMZ, Prediction, KMZWarning
from .basins import find_basins, BasinResult
from .analysis import analyze, AnalysisReport
from .evaluation import DictionaryEvaluation

__version__ = "0.2.2"
__all__ = ["Data", "KMZ", "Prediction", "KMZWarning", "find_basins", "BasinResult", "analyze",
           "AnalysisReport", "DictionaryEvaluation"]
