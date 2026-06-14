"""api: the inference boundary. Loads the trained KAN + scaler and exposes
predict() and the geographic risk scoring used by the dashboard."""

from api.predict import predict, predict_from_features
from api.geo import strategic_value_grid, synthesize_ground_path, risk_heatmap

__all__ = [
    "predict",
    "predict_from_features",
    "strategic_value_grid",
    "synthesize_ground_path",
    "risk_heatmap",
]
