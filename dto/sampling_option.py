from dataclasses import dataclass


@dataclass
class SamplingOption:
    """Sampling options for sampling images and lidar files.
    """
    offset: int = 25
    ratio: int = 1  # multiples of 0.1
