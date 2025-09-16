# raw data utils for manage rawdata function.
import os
import re
from typing import Set, Dict
from enum import Enum

RAW_FOLDER_PATTERN = '(\\S*)\\d{8}_\\d{6}(\\S*)$'
RAW_FOLDER_RE = re.compile(RAW_FOLDER_PATTERN)
DEFAULT_DEPTH = 5

class RawFileExtension(Enum):
    """
    Enum class for raw file.
    """
    LIDAR = '.tar'
    VIDEO = '.h264'
    FRAME = '.txt'
    CAMINFO = '.json'

def is_hidden_file_on_unix_like_system(file_path: str) -> bool:
    """If the file is hidden(starts with .) return True otherwise return False.

    Args:
        file_path (str): The file's path.

    Returns:
        bool: The file is hidden or not.
    """
    # On Unix-like systems, check if the file name starts with a dot

    return any(file_name.startswith('.')
               for file_name in file_path.split(os.path.sep))

def search_rawdata_dirs_from(search_dir: str, max_depth: int = DEFAULT_DEPTH) -> Set[str]:
    """
    Search for surf rawdata directories from the given search directory.

    Args:
        search_dir (str): The directory to start the search from.
        max_depth (int, optional): The maximum depth to search. Defaults to DEFAULT_DEPTH.

    Returns:
        Set[str]: A set of surf rawdata directories found during the search.
    """
    rawdata_set: Set[str] = set()

    def recursive_search(current_dir: str, current_depth: int):
        if current_depth > max_depth:
            return

        if not is_hidden_file_on_unix_like_system(current_dir):
            for entry in os.scandir(current_dir):
                if entry.is_dir(follow_symlinks=False):
                    if RAW_FOLDER_RE.match(entry.name):
                        rawdata_set.add(entry.path)
                    recursive_search(entry.path, current_depth + 1)

    recursive_search(search_dir, 0)
    if not rawdata_set:
        if RAW_FOLDER_RE.match(search_dir):
            rawdata_set.add(search_dir)

    return rawdata_set

def get_cam_info_file(rawdata_dir: str, cam_num: int) -> str:
    """
    Get the cam info file path from the given rawdata directory.

    Args:
        rawdata_dir (str): The rawdata directory.
        cam_num (int): The camera number.

    Returns:
        str: The cam info file path.
    """
    cam_info_name = f'cam_info_cam_{cam_num}' if cam_num != 0 else 'cam_info_cam_aptive'

    for entry in os.scandir(rawdata_dir):
        if entry.is_file() and cam_info_name in entry.name and entry.name.endswith(RawFileExtension.CAMINFO.value):
            return entry.path

    # seek once again when we don't find the cam info file with the given cam number.
    if cam_num == 0:
        for entry in os.scandir(rawdata_dir):
            if entry.is_file() and entry.name.endswith(RawFileExtension.CAMINFO.value):
                return entry.path

    return ''

# def get_sampling_required_files(raw: str, cam_num: int) -> Dict[RawFileExtension, str]:
#     """
#     Get the format file from the given raw file.

#     Returns (Tuple[str, str])
#         the format file path.
#     """
#     sampling_required_files = {}
#     if not os.path.isdir(raw):
#         return

#     raw_files = os.listdir(raw)
#     for file in raw_files:
#         for ext in RawFileExtension:
#             if file.endswith(ext.value):
#                 if cam_num and f'cam_{cam_num}' in file:
#                     sampling_required_files[ext.name] = os.path.join(raw, file)
#                 else:


#     return sampling_required_files


def is_aptive_video(file_name: str) -> bool:
    """
    Check if the given file is an aptive video file.

    Args:
        file_name (str): The file name to check.

    Returns:
        bool: True if the file is an aptive video file, False otherwise.
    """
    pattern = re.compile(r'.*\d{3}\.h264$')
    return file_name.lower().endswith('cam.h264') or bool(pattern.match(file_name))

def get_sampling_video(raw: str, cam_num: int) -> str:
    """
    Get the video file from the given raw file.

    Returns
        the video file path.
    """
    if not os.path.isdir(raw):
        return

    raw_files = os.listdir(raw)
    for file in raw_files:
        if file.endswith(RawFileExtension.VIDEO.value):
            if cam_num:
                if f'cam_{cam_num}' in file:
                    return os.path.join(raw, file)
            elif cam_num == 0:
                if is_aptive_video(file):
                    return os.path.join(raw, file)
    return ''

def get_sampling_txt(video_file: str) -> str:
    """
    Get the txt file from the given raw file.

    Returns
        the txt file path.
    """
    if not os.path.exists(video_file):
        return

    txt_file = video_file.replace(RawFileExtension.VIDEO.value, RawFileExtension.FRAME.value)
    if os.path.exists(txt_file):
        return txt_file

    return ''


def get_sampling_lidar(raw: str) -> str:
    if not os.path.isdir(raw):
        return

    raw_files = os.listdir(raw)
    for file in raw_files:
        if file.lower().endswith(RawFileExtension.LIDAR.value) and 'lidar' in file.lower():
            return os.path.join(raw, file)

    return ''
