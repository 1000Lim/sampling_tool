import os
import cv2
from typing import Dict, List
from dto.sample_frame import SampleFrame
from tools.lidar_to_cam_matcher import LidarMatchedCamFrame
IMAGE_EXTENSION = '.png'

def sampling_target_frames_from_video(video_path: str,
                                      sampling_frames: List[SampleFrame],
                                      result_dir: str,
                                      ignore_frames: List[int]
                                      ):
    """Writes images from the video.

    Args:
        video_path (str): path of the video.
        tag_name(str): the prefix name added on the video.
        sampling_frames (Dict): Frames to extract.
            - key: frame_index.
            - value: lidar's timestamp.
        result_dir (str): path of the sampling files.
        ignore_frames: ignore frames to not extract(Aptive Cams have some frames that are not useful for the sampling process.)
    """
    generated_frames = []
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    os.makedirs(result_dir, exist_ok=True)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for sample_frame in sampling_frames:
            if frame_index == sample_frame.frame_id:
                if ignore_frames and frame_index in ignore_frames:
                    continue
                sample_frame.name = f"{video_name}_{sample_frame.lidar_timestamp}_{frame_index:06d}"
                file_name = sample_frame.name + IMAGE_EXTENSION
                generated_frames.append(file_name)
                file_path = os.path.join(result_dir, file_name)
                cv2.imwrite(file_path, frame)
        frame_index += 1
    cap.release()

    return generated_frames


def get_sample_frames(lidar_timestamps: List[int], offset: int, frame_file) -> List[SampleFrame]:
    """Get the sample frames that has nearest timestamp from the frame file.
    """
    if not os.path.isfile(frame_file):
        raise FileNotFoundError(f'frame file: {frame_file} not found.')

    frame_dict = dict()
    sample_frames: List[SampleFrame] = []

    with open(frame_file, 'r', encoding='utf-8') as frame_text:
        lines = frame_text.readlines()
        for line in lines:
            index, timestamp = line.rstrip('\n').split(',')
            frame_dict[int(index)] = int(timestamp)

    for timestamp in lidar_timestamps:
        min_key = min(frame_dict, key=lambda k: abs(offset * 1000 + timestamp - frame_dict[k]))
        new_frame = SampleFrame(
            frame_id=min_key,
            frame_timestamp=frame_dict[min_key],
            lidar_timestamp=timestamp
        )
        sample_frames.append(new_frame)

    sorted(sample_frames, key=lambda x: x.frame_id)
    return sample_frames


def get_timestamps_from_frame_file(frame_file: str):
    """
    Retrieves timestamps from a frame file.

    Args:
        frame_file (str): The path to the frame file.

    Returns:
        dict: A dictionary containing the timestamps, where the keys are the frame indices and the values are the timestamps.

    Raises:
        FileNotFoundError: If the frame file is not found.
    """
    frame_dict = dict()
    if not os.path.isfile(frame_file):
        raise FileNotFoundError(f'frame file: {frame_file} not found.')

    with open(frame_file, 'r', encoding='utf-8') as frame_text:
        lines = frame_text.readlines()
        for line in lines:
            index, timestamp = line.rstrip('\n').split(',')
            frame_dict[int(index)] = int(timestamp)

    return frame_dict
