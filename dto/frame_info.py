from dataclasses import dataclass, field
from typing import List


@dataclass
class ChannelImage:
    """Image timestamp.
    """
    image_timestamp: int
    lidar_timestamp: int
    camera_key: str
    image_index: int
    image_path: str = ''
    webp_path: str = ''


@dataclass
class LidarInfo:
    """Lidar timestamp
    """
    name: str
    path: str
    type: str = "Original"


@dataclass
class FrameInfo:
    """Frame information.
    """
    frame_name: str
    dataset_id: str
    rawdata_id: str
    ffc_timestamp: int
    lidar_timestamp: int
    ffc_index: int
    tags: List[str] = field(default_factory=list)
    channel_images: List[ChannelImage] = field(default_factory=list)
    lidar: LidarInfo = field(default_factory=LidarInfo)
