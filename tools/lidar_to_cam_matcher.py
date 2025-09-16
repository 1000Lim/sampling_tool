"""
This script finds timestamp-based LiDAR PCD to camera image matching indices.
"""
import math
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from dto.circular_lidar import CircularLidarConfig, Pandar64
from common.calibration import Calibration
from util.coordinate_util import ics_to_vcs_points, vcs_to_ics_points


CAM_DEFAULT_EXPOSURE_TIME = 1000000//30  # microsecond
INVALID_MATCH = -1

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



def calculate_motor_angle_range(calib: Calibration,
                                circular_lidar_config: CircularLidarConfig = Pandar64):
    """
    Calculates the motor angle range that matches the field of view (FOV) of the camera to the circular lidar.
    The actual FOV of camera can't be greater than 180, because camera is blocked by vehicle body.

    Args:
        calib (Calibration): The calibration object containing the camera parameters.
        circular_lidar_config (CircularLidarConfig, optional): The configuration of the circular lidar.

    Returns:
        Tuple[float, float]: A tuple containing the motor angle range (begin degree and end degree)
        that matches the FOV of the camera to the circular lidar.
        begin degree must be smaller than end degree. (ex 350~10 -> -10~10)
    """

    img_w = calib.width
    img_h = calib.height

    # get VCS of center point.
    # Center point to ensure VCS point is visble to camera
    center_coord = np.array([[calib.cx, calib.cy, 0]], dtype=np.float32)
    center_vcs_coord = ics_to_vcs_points(center_coord, calib)
    center_z = center_vcs_coord[0, 2] # get height offset of camera from lidar
    center_distance_at_z = math.sqrt(center_vcs_coord[0, 0]**2 + center_vcs_coord[0, 1]**2) # get distance from lidar to camera

    # start from 1 meter from center point
    left_degree_list = []
    right_degree_list = []
    min_distance = math.ceil(center_distance_at_z)
    for distance in range(min_distance, circular_lidar_config.max_distance, circular_lidar_config.check_depth):
        # constructing coord of circle(center is lidar) on z plane of camera
        begin_degree = 0
        end_degree = 360
        inc_degree = 0.01
        degree_arr = np.arange(begin_degree, end_degree, inc_degree)
        x_arr = +np.sin(np.radians(degree_arr)) * distance
        y_arr = +np.cos(np.radians(degree_arr)) * distance
        z_arr = np.full_like(x_arr, center_z)
        circle_vcs_points = np.stack([x_arr, y_arr, z_arr], axis=1)
        ics_coord = vcs_to_ics_points(circle_vcs_points, calib)
        in_range = (
            (ics_coord[:, 0] > 0) &
            (ics_coord[:, 0] < img_w) &
            (ics_coord[:, 1] > 0) &
            (ics_coord[:, 1] < img_h) &
            (ics_coord[:, 2] > 0) # this condition will discard ray's angle is greater than 180
        )
        # idx list has True value
        true_idx_list = np.where(in_range)[0]
        assert len(true_idx_list) > 0, "No matching angle range"
        assert len(true_idx_list) < len(degree_arr), "All angle range is matching"
        if in_range[0] and in_range[-1]: # circular(-1~0) case
            false_idx_list = np.where(~in_range)[0]
            left_idx = false_idx_list[-1] + 1
            right_idx = false_idx_list[0] - 1
            assert left_idx > right_idx, "left_idx must be smaller than right_idx in circular case"
        else:
            left_idx = true_idx_list[0]
            right_idx = true_idx_list[-1]
        # to degree
        left_degree = left_idx*inc_degree + begin_degree
        right_degree = right_idx*inc_degree + begin_degree
        #print(f'distance: {distance}, left_degree: {left_degree:.2f}, right_degree: {right_degree:.2f}')
        left_degree_list.append(left_degree)
        right_degree_list.append(right_degree)
    left_degree_list_np = np.array(left_degree_list)
    right_degree_list_np = np.array(right_degree_list)
    # circular case [270, 360], [0 , 90]
    if left_degree_list_np.min() < 90 and  left_degree_list_np.max() > 270:
        left_degree_list_np = np.where(left_degree_list_np > 270, left_degree_list_np - 360, left_degree_list_np)
    if right_degree_list_np.min() < 90 and  right_degree_list_np.max() > 270:
        right_degree_list_np = np.where(right_degree_list_np > 270, right_degree_list_np - 360, right_degree_list_np)
    left_most = float(left_degree_list_np.min())
    right_most = float(right_degree_list_np.max())
    if left_most > right_most:
        left_most -= 360
    # print(f'left {min(left_degree_list):.2f}~{max(left_degree_list):.2f}, right {min(right_degree_list):.2f}~{max(right_degree_list):.2f}')
    # print(f'Lidar FOV : {left_most:.2f} ~ {right_most:.2f}')
    return left_most, right_most


def find_best_matching_cam_frame_from_circular_lidar(
        motor_angle_begin: float,
        motor_angle_end: float,
        lidar_timestamp: int,
        camera_timestamps: Dict,
        circular_lidar_config: CircularLidarConfig = Pandar64,
        camera_expousre_time: int = CAM_DEFAULT_EXPOSURE_TIME) -> LidarMatchedCamFrame:
    """
    Finds the best matching camera frame for a given circular lidar scan.

    Args:
        motor_angle_begin (float): The starting motor angle of the circular lidar scan.
        motor_angle_end (float): The ending motor angle of the circular lidar scan.
        lidar_timestamp (int): The timestamp of the lidar scan.
        camera_timestamps (Dict): A dictionary of camera indices and their corresponding timestamps.
        lidar_config (CircularLidarConfig, optional): The configuration of the circular Lidar.
            Defaults to Pandar64.
        camera_expousre_time (int, optional): The exposure time of the camera.
            Defaults to CAM_DEFAULT_EXPOSURE_TIME.

    Returns:
        MatchedFrame: The best matching camera frame.

    """
    best_matching_frame = None
    max_matching_timestamp = 0

    angle_time_begin = circular_lidar_config.mortol_speed * motor_angle_begin
    angle_time_end = circular_lidar_config.mortol_speed * motor_angle_end

    for camera_index, camera_timestamp in camera_timestamps.items():
        camera_fsync_begin = camera_timestamp - camera_expousre_time
        camera_fsync_end = camera_timestamp

        motor_start_time = lidar_timestamp + int(angle_time_begin)
        motor_end_time = lidar_timestamp + int(angle_time_end)

        if camera_fsync_begin <= motor_start_time <= motor_end_time <= camera_fsync_end:
            return LidarMatchedCamFrame(camera_index=camera_index,
                                lidar_timestamp=lidar_timestamp,
                                camera_fsync_begin=camera_fsync_begin,
                                camera_fsync_end=camera_fsync_end,
                                motor_fov_begin=motor_start_time,
                                motor_fov_end=motor_end_time,
                                overlap_begin=motor_start_time,
                                overlap_end=motor_end_time,
                                match_percentage=1.0)

        overlap_begin = max(motor_start_time, camera_fsync_begin)
        overlap_end = min(motor_end_time, camera_fsync_end)

        if overlap_end - overlap_begin > max_matching_timestamp:
            max_matching_timestamp = overlap_end - overlap_begin
            mactched_percentage = max_matching_timestamp / camera_expousre_time
            best_matching_frame = LidarMatchedCamFrame(
                camera_index=camera_index,
                lidar_timestamp=lidar_timestamp,
                camera_fsync_begin=camera_fsync_begin,
                camera_fsync_end=camera_fsync_end,
                motor_fov_begin=motor_start_time,
                motor_fov_end=motor_end_time,
                overlap_begin=overlap_begin,
                overlap_end=overlap_end,
                match_percentage=mactched_percentage)

    if not best_matching_frame:
        best_matching_frame = LidarMatchedCamFrame(
            camera_index=INVALID_MATCH,
            lidar_timestamp=lidar_timestamp,
            camera_fsync_begin=INVALID_MATCH,
            camera_fsync_end=INVALID_MATCH,
            motor_fov_begin=motor_start_time,
            motor_fov_end=motor_end_time,
            overlap_begin=INVALID_MATCH,
            overlap_end=INVALID_MATCH,
            match_percentage=0)

    return best_matching_frame


def write_export_interface_file(matched_frames: List[LidarMatchedCamFrame], export_interface_file: str | None):
    """
    Writes the matched frames to an export interface file.

    Args:
        matched_frames (List[MatchedFrame]): List of matched frames.
        export_interface_file (str): The path to the export interface file.

    The file should be formatted as follows:
    # https://stradvision.atlassian.net/browse/APR-207
    # (lidar frame index) (lidar frame timestamp) (camera frame index)
    # (camera fsync begin timestamp) (camera fsync end timestamp)
    # (overlap begin timestamp) (overlay end timestamp)
    """
    lidar_frame_index = 0
    lines = []

    # write the comments
    lines.append("# (lidar frame index) (lidar frame timestamp) (camera frame index) "
                 "(camera fsync begin timestamp) (camera fsync end timestamp) "
                 "(overlap begin timestamp) (overlay end timestamp)\n")
    for match_frame in matched_frames:
        line_info = (
            f"{lidar_frame_index} {match_frame.lidar_timestamp} {match_frame.camera_index} "
            f"{match_frame.camera_fsync_begin} {match_frame.camera_fsync_end} "
            f"{match_frame.overlap_begin} {match_frame.overlap_end}\n"
        )
        lidar_frame_index += 1
        lines.append(line_info)

    if not export_interface_file:
        return
    try:
        with open(export_interface_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception:
        # best-effort; do not fail pipeline because export path is invalid
        pass


def find_matching_cam_frames_from_circular_lidar(
        motor_begin_angle: float,
        motor_end_angle: float,
        lidar_timestamps: List[int],
        camera_timestamps: Dict,
        matching_threthold: float = 0,
        circular_lidar_config: CircularLidarConfig = Pandar64,
        export_interface_file: str | None = None,
        camera_expousre_time: int = CAM_DEFAULT_EXPOSURE_TIME) -> List[LidarMatchedCamFrame]:
    """
    Finds the matching camera frames from a circular lidar scan.

    Args:
        motor_begin_angle (float): The starting angle of the motor.
        motor_end_angle (float): The ending angle of the motor.
        lidar_timestamps (List[int]): List of lidar timestamps.
        camera_timestamps (Dict): Dictionary of camera timestamps.
        matching_threthold (float, optional): The matching threshold percentage. Defaults to 0.
        circular_lidar_config (CircularLidarConfig, optional):
            The configuration of the circular lidar. Defaults to Pandar64.
        export_interface_file (str, optional):
            The path to the interface file to export the matched frames. Defaults to None.
        camera_expousre_time (int, optional): The exposure time of the camera.
            Defaults to CAM_DEFAULT_EXPOSURE_TIME.

    Returns:
        List[MatchedFrame]: List of matched frames.
    """

    matched_frames = []
    for lidar_timestamp in lidar_timestamps:
        best_matched_frame = find_best_matching_cam_frame_from_circular_lidar(
            motor_angle_begin=motor_begin_angle,
            motor_angle_end=motor_end_angle,
            lidar_timestamp=lidar_timestamp,
            camera_timestamps=camera_timestamps,
            circular_lidar_config=circular_lidar_config,
            camera_expousre_time=camera_expousre_time
        )

        if best_matched_frame and best_matched_frame.match_percentage >= matching_threthold:
            matched_frames.append(best_matched_frame)

    # sort the matched frames by lidar timestamp.
    sorted(matched_frames, key=lambda x: x.lidar_timestamp)

    # Write only if a valid path is explicitly provided
    if export_interface_file:
        write_export_interface_file(matched_frames, export_interface_file)

    return matched_frames
