from __future__ import annotations

__doc__ = """A package for performing bin merging in an optimal fashion to preserve
seperability between N hypotheses. Metrics such as the ROC and LOC metrics are provided,
as well as merging tools in MergerLocal and MergerNonlocal.
"""
__name__ == "MiLoMerge"
__version__ = "1.0.0"
__author__ = "Mohit V. Srivastav, Michalis Panagiotou, Lucas S. Kang"
from .merging.bin_optimizer import mlm_driver as mlm
from .merging.bin_optimizer import MergerLocal, MergerNonlocal
from .metrics.ROC_curves import ROC_curve
from .metrics.ROC_curves import LOC_curve
from .merging.place_from_map import place_array_nonlocal
from .merging.place_from_map import place_event_nonlocal
from .merging.place_from_map import place_local

__all__ = [
    "mlm",
    "MergerLocal",
    "MergerNonlocal",
    "ROC_curve",
    "LOC_curve",
    "place_array_nonlocal",
    "place_event_nonlocal",
    "place_local"
]
