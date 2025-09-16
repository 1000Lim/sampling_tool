"""This module is made for timestamp matching between lidar and camera images.
"""

import os
import json
from loguru import logger
from util import lidar_util, cam_util

FFC_FRAME_SUFFIX = '_cam_1.txt'
FFC_MARK = '2160P_H120_FFC'


def get_ffc_cam_name(cam_info_json: str) -> str:
    """This method will get the ffc camera index from the camera info json file.
    Args:
        cam_info_json (str): The camera info json file.

    Returns:
        int: The ffc camera index.
    """
    if not os.path.isfile(cam_info_json):
        logger.error(f"Can not find the camera info json file:{cam_info_json}")
        return

    with open(cam_info_json, 'r') as f:
        cam_info = json.load(f)

    for cam in cam_info['camera_list']:
        if 'height' in cam and 'hfov' in cam and 'pos' in cam and 'camera' in cam:
            if cam['height'] == 2160 and cam['hfov'] == 120 and cam['pos'].upper() == 'FFC':
                return cam['camera']
    return None


def generate_mathed_meta_json(raw_sampled_dir: str):
    """This method will generate the matched meta data for the raw sampled directory.
    Args:
        rawdata_sampled_dir (str): The raw-data directory that already sampled.

    Returns:
        JSON: The matched meta data.
    """
    if not os.path.isdir(raw_sampled_dir):
        logger.error(f"Can not find the raw directory:{raw_sampled_dir}")
        return

    # grep timestamps from the sampled lidar files.
    lidar_dir = os.path.join(raw_sampled_dir, 'lidar')
    if not os.path.isdir(lidar_dir):
        logger.error(f"Can not find the lidar directory:{lidar_dir}")
        return

    lidar_files = [file for file in os.listdir(
        lidar_dir) if file.endswith('.pcd')]
    if not lidar_files:
        logger.error(
            f"Can not find the lidar files in the directory:{lidar_dir}")
        return

    lidar_timestamps = [int(lidar_util.get_lidar_timestamps_from_pcd(
        file)) for file in lidar_files]

    frame_file = None
    cam_info_files = [file for file in os.listdir(
        raw_sampled_dir) if file.endswith('_cam_info.json')]

    for cam_info in cam_info_files:
        ffc_cam = get_ffc_cam_name(os.path.join(raw_sampled_dir, cam_info))
        if ffc_cam:
            break

    if not ffc_cam:
        logger.error(
            "Can not find the FFC camera in the camera info json file.")
        return

    for file in os.listdir(raw_sampled_dir):
        if ffc_cam.upper() in file.upper() and file.endswith('.txt'):
            frame_file = os.path.join(raw_sampled_dir, file)
            break

    if not frame_file:
        logger.error(
            f"Can not find the frame file in the directory:{raw_sampled_dir}")
        return

    frames = cam_util.get_sample_images_with_lidar_timestamps(
        lidar_timestamps, 50, frame_file, '2160p_h120_ffc')

    if not frames:
        logger.error("Can not find the matched frames.")
        return

    sampled_list = []
    for frame in frames:
        sampled_list.append({
            'lidar_timestamp': frame.lidar_timestamp,
            'front_timestamp': frame.image_timestamp,
            'front_index': frame.image_index,
            'diff_timestamp': frame.lidar_timestamp - frame.image_timestamp
        })

    sampled_list.sort(key=lambda x: x['front_index'])

    # Specify the filename to write the JSON data
    matched_json = os.path.join(raw_sampled_dir, 'matched.json')

    # Open a file in write mode
    with open(matched_json, 'w') as f:
        # Write the dictionary to file as JSON
        json.dump({
            'sample_json': sampled_list
        }, f, indent=4)

    return matched_json
