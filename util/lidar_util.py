"""
Lidar util supports modules for lidar files.
"""
import os
import numpy as np
from typing import List
import tarfile
from pypcd4 import PointCloud
from common.calibration import Calibration
from util.coordinate_util import vcs_to_ics_points
LIDAR_EXTENSION = '.pcd'
EXTRACTED_LIDAR_EXTENSION = '.tar'
LIDAR_ROTATION_TIME = 100 * 1000  # 100 ms
DEFAULT_SAMPLING_DIR = 'point_clouds'
FRAME_FILE_EXTENSION = '.txt'
SAMPLED_PREFIX = 'sampled_'
LIDAR_TIMESTAMP_REGEX = r'17\d{14}' # Regular expression to match the pattern for getting lidar timestamp from filename.


def get_timestamps_from_extracted_tar(extracted_tar: str, sampling_ratio: int = 1) -> List[int]:
    """Gets the nearest frames from the extracted_tar

    Args:
        extracted_tar (str): extracted tar file.
    Returns:
        List of the timestamps included in the file.
    """
    timestamps = []

    if not os.path.isfile(extracted_tar):
        raise FileNotFoundError(f'lidar extracted tar file: {extracted_tar} not found.')

    if not extracted_tar.lower().endswith(EXTRACTED_LIDAR_EXTENSION):
        error_msg = f'The lidar file {extracted_tar} is not ends with {EXTRACTED_LIDAR_EXTENSION}'
        raise NameError(error_msg)

    # return the lidar file from the extracted tar file.
    with tarfile.open(extracted_tar, 'r') as archive:
        for file_name in archive.getnames():
            if file_name.endswith(LIDAR_EXTENSION):
                timestamp = os.path.basename(file_name).split(LIDAR_EXTENSION)[0]
                timestamps.append(int(timestamp))

    result = sorted(timestamps)[::sampling_ratio]
    return result


def copy_sampled_lidar(extracted_tar: str, sampling_ratio: int = 1,
                       copy_dir: str = DEFAULT_SAMPLING_DIR):
    """copy sampling lidar from files to the target folder.

    Args:
        lidar_files (List[float]): write sampling lidar files on target folder.
    """
    sampling_timestamps = get_timestamps_from_extracted_tar(extracted_tar, sampling_ratio)


    os.makedirs(copy_dir, exist_ok=True)

    with tarfile.open(extracted_tar, 'r') as archive:
        for file in archive.getnames():
            if file.endswith(LIDAR_EXTENSION):
                timestamp = os.path.basename(file).split(LIDAR_EXTENSION)[0]
                if float(timestamp) in sampling_timestamps:
                    if os.path.dirname(file) == 'lidar':
                        archive.extract(file, copy_dir)
                    else:
                        archive.extract(file, os.path.join(copy_dir, 'lidar'))

    return sampling_timestamps

def extract_target_lidar_files(extracted_tar: str, target_pcds: List[int], copy_dir: str = DEFAULT_SAMPLING_DIR):
    """Extract the target lidar files from the extracted tar file."""
    extracted_timestamps = []
    if not os.path.isdir(copy_dir):
        os.makedirs(copy_dir, exist_ok=True)

    with tarfile.open(extracted_tar, 'r') as archive:
        for file in archive.getnames():
            if os.path.basename(file) in target_pcds:
                timestamp = os.path.basename(file).split(LIDAR_EXTENSION)[0]
                extracted_timestamps.append(int(timestamp))
                archive.extract(file, copy_dir)

    return extracted_timestamps

def rename_sampled_lidar(image_names: List, copy_dir: str = DEFAULT_SAMPLING_DIR):
    """Rename the sampled lidar files.

    Args:
        basename (str): the basename of the file.
        copy_dir (str): the directory to rename the files.
    """
    for file in os.listdir(copy_dir):
        if file.endswith(LIDAR_EXTENSION):
            file_name = os.path.splitext(file)[0]
            for image in image_names:
                if file_name in image:
                    image_name = os.path.splitext(os.path.basename(image))[0]
                    new_name = image_name + LIDAR_EXTENSION
                    os.rename(os.path.join(copy_dir, file), os.path.join(copy_dir, new_name))
                    break

def read_pcd_file(file_path:str):
    pc_obj = PointCloud.from_path(file_path)
    point_cloud_array = np.hstack([pc_obj.pc_data['x'][:, np.newaxis], pc_obj.pc_data['y'][:, np.newaxis], pc_obj.pc_data['z'][:, np.newaxis], pc_obj.pc_data['intensity'][:, np.newaxis]])
    return point_cloud_array


def get_lidar_points_in_image(lidar_points:np.ndarray, calib:Calibration, image_size:tuple):
    """Get the lidar points in the image."""
    lidar_points_in_ics = vcs_to_ics_points(lidar_points[:, :3], calib)
    in_image_mask = (lidar_points_in_ics[:, 0] >= 0) & (lidar_points_in_ics[:, 0] < image_size[0]) & (lidar_points_in_ics[:, 1] >= 0) & (lidar_points_in_ics[:, 1] < image_size[1]) & (lidar_points_in_ics[:, 2] >= 0)
    lidar_points_in_ics = lidar_points_in_ics[in_image_mask]
    return lidar_points_in_ics, lidar_points[in_image_mask]


def get_timestamp_info_from_lidar(file_path:str):
    """Get the min timestamp in the pcd file(min, max)."""
    pc_obj = PointCloud.from_path(file_path)
    return pc_obj.pc_data['timestamp'].min(), pc_obj.pc_data['timestamp'].max()


def get_angle_from_image_to_lidar(image_timestamp: int, lidar_timestamp: int):
    """Get the angle from image to lidar."""
    return (image_timestamp - lidar_timestamp) / LIDAR_ROTATION_TIME * 360
