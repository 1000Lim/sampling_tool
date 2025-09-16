"""This module supports the rawdata related features.
"""
from dataclasses import dataclass, field
from typing import List
from consts import Status


@dataclass
class RawVideo:
    cam_pos: str = ""
    cam_resolution: str = ""
    video_path: str = ""
    frame_path: str = ""
    camera_key: str = ""
    calibration: dict = field(default_factory=dict)


@dataclass
class RawData:
    """Information that rawdata tooks on.
    """
    rawdata_id: str = ""
    data_key: str = ""
    path: str = ""
    videos: List[RawVideo] = field(default_factory=list)
    lidar_path: str = ""
    offset: int = 0
    ratio: float = 0  # multiples of 0.1


@dataclass
class RawdataInfo:
    """Rawdata information.
    """
    rawdata_key: str = ''
    path: str = ''
    update_time: str = ''
    tags: List[str] = field(default_factory=list)
    is_sampled: bool = False
    sampling_job_status = Status.UNPROCESSED.value
