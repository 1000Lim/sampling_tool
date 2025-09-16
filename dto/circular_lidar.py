"""The script manage circular lidar configuration.
"""
from dataclasses import dataclass

PANDAR_64_MARGIN_OFFSET = 5.208


@dataclass
class CircularLidarConfig:
    """Circular lidar configuration."""
    mortol_speed: float
    margin_offset_degree: float
    channels: int
    min_distance: int
    max_distance: int
    check_depth: int


Pandar64 = CircularLidarConfig(
    mortol_speed=100000 // 360,  # 100000 us / 360 degree
    margin_offset_degree=PANDAR_64_MARGIN_OFFSET,
    channels=64,
    min_distance=10,   # meter
    max_distance=400,  # meter
    check_depth=10,  # meter
)
