"""IMU double-tap recognition package."""

from .model import CausalCNNGRU
from .physics import IMUSimulator, TapParams
from .inference import OnlineDoubleTapDetector
from .recording import load_imu_csv, load_labels, export_labeled_dataset

__all__ = [
    "CausalCNNGRU",
    "IMUSimulator",
    "TapParams",
    "OnlineDoubleTapDetector",
    "load_imu_csv",
    "load_labels",
    "export_labeled_dataset",
]
