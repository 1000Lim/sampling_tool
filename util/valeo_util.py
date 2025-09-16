import os
from enum import Enum
from loguru import logger


class ValeoFileExtension(Enum):
    """
    Enum class for raw file.
    """
    LIDAR = '.tar'
    IMAGE_TAR = '.tar'
    FRAME = '.txt'
    CAMINFO = '.json'



def get_valeo_sampling_required_files(dir_path: str) -> str:
    """
    Get the video file from the given raw file.

    Returns
        the video file path.
    """
    if not os.path.isdir(dir_path):
        logger.error(f"Error: The directory {dir_path} does not exist.")
        return


    files = os.listdir(dir_path)
    image_tar_file = ''
    frame_txt_file = ''
    cam_info_file = ''
    lidar_tar_file = ''
    for file in files:
        if file.endswith(ValeoFileExtension.IMAGE_TAR.value) and 'cam_valeo' in file:
            image_tar_file = os.path.join(dir_path, file)
        elif file.endswith(ValeoFileExtension.FRAME.value) and 'cam_valeo' in file:
            frame_txt_file = os.path.join(dir_path, file)
        elif file.endswith(ValeoFileExtension.CAMINFO.value) and 'cam_valeo' in file:
            cam_info_file = os.path.join(dir_path, file)
        elif file.endswith(ValeoFileExtension.LIDAR.value) and 'cam_valeo' not in file:
            lidar_tar_file = os.path.join(dir_path, file)
    return image_tar_file, frame_txt_file, cam_info_file, lidar_tar_file
    