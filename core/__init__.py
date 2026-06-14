"""core: the single source of truth for the UAV intent estimation project.

Every other module (data generator, model, api, app) imports definitions from
here. Nothing duplicates a class label, a parameter name, the actuator envelope,
or the feature-extraction logic. Change it once, the whole system stays coherent.
"""

from core.config import (
    SEED,
    Intent,
    INTENT_NAMES,
    INTENT_COLORS,
    PARAMETERS,
    TRACK_LENGTH,
    TORQUE_ENVELOPE_PCT,
)
from core.features import FEATURE_NAMES, extract_features

__all__ = [
    "SEED",
    "Intent",
    "INTENT_NAMES",
    "INTENT_COLORS",
    "PARAMETERS",
    "TRACK_LENGTH",
    "TORQUE_ENVELOPE_PCT",
    "FEATURE_NAMES",
    "extract_features",
]
