import os
from enum import Enum
from loguru import logger


class AclmFileExtension(Enum):
    """
    Enum class for ACLM raw file.
    """
    LIDAR = '.tar'
    IMAGE_TAR = '.tar'
    FRAME = '.txt'
    CAMINFO = '.json'



def get_aclm_sampling_required_files(dir_path: str) -> tuple:
    """
    Get the required files for ACLM sampling from the given directory.

    Args:
        dir_path: Directory path containing ACLM raw data

    Returns:
        tuple: (image_tar_file, frame_txt_file, cam_info_file, lidar_tar_file)
    """
    if not os.path.isdir(dir_path):
        logger.error(f"Error: The directory {dir_path} does not exist.")
        return ('', '', '', '')


    files = os.listdir(dir_path)
    image_tar_file = ''
    frame_txt_file = ''
    cam_info_file = ''
    lidar_tar_file = ''

    for file in files:
        # Image tar: cam_aclm.tar
        if file.endswith(AclmFileExtension.IMAGE_TAR.value) and 'cam_aclm.tar' in file:
            image_tar_file = os.path.join(dir_path, file)
        # Frame txt: cam_aclm.txt
        elif file.endswith(AclmFileExtension.FRAME.value) and 'cam_aclm.txt' in file:
            frame_txt_file = os.path.join(dir_path, file)
        # Cam info: cam_info_cam_aclm.json
        elif file.endswith(AclmFileExtension.CAMINFO.value) and 'cam_info_cam_aclm.json' in file:
            cam_info_file = os.path.join(dir_path, file)
        # LiDAR tar: any .tar file that doesn't contain 'cam_aclm'
        elif file.endswith(AclmFileExtension.LIDAR.value) and 'cam_aclm' not in file and 'lidar' in file:
            lidar_tar_file = os.path.join(dir_path, file)

    return image_tar_file, frame_txt_file, cam_info_file, lidar_tar_file
