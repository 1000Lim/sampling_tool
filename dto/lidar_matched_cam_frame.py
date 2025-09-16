from dataclasses import dataclass


@dataclass
class LidarMatchedCamFrame:
    """dataclass for the matched frame.
    """
    camera_index: int  # the camera index.
    lidar_timestamp: int  # the lidar timestamp.
    camera_fsync_begin: int  # camera fsync begin timestamp.
    camera_fsync_end: int  # camera fsync end timestamp.
    motor_fov_begin: int  # motor fov begin timestamp.
    motor_fov_end: int   # motor fov end timestamp.
    overlap_begin: float  # the matched begin angle.
    overlap_end: float  # the matched end angle.
    match_percentage: float  # the matched percentage.
