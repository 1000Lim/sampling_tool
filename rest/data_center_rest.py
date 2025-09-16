"""REST API that are communicating with the data center.
"""
import os
from typing import List
from urllib.parse import urljoin
import requests
from loguru import logger
from dto.rawdata_info import RawData, RawVideo, RawdataInfo
from dto.frame_info import FrameInfo
from consts import (TIMEOUT, DATACENTER_RAW_DATA_URL,
                    DATACENTER_DATASET_URL, DATACENTER_FRAME_URL, Status)


def get_rawdata_list():
    """Gets the rawdata list archived in data center.
    """
    rawdata_ids: List[RawdataInfo] = []

    def _append_rawdata_if_has_next(page: int):
        rawdata_list_url = urljoin(DATACENTER_RAW_DATA_URL, 'raw-datas')
        params = {'page': page}
        response = requests.get(rawdata_list_url,
                                params=params,
                                timeout=TIMEOUT,
                                headers={"Content-Type": "application/json"}
                                )
        if response.status_code == 200:
            if 'docs' in response.json():
                for doc in response.json()['docs']:
                    rawdata_info = RawdataInfo()
                    if 'dataKey' in doc:
                        rawdata_info.rawdata_key = doc['dataKey']
                        rawdata_info.path = doc['path']
                        rawdata_info.update_time = doc['updatedAt']
                        rawdata_info.tags = doc['tags']
                        if 'jobStatus' in doc and 'Sampling' in doc['jobStatus']:
                            rawdata_info.sampling_job_status = doc['jobStatus']['Sampling']

                    rawdata_ids.append(rawdata_info)
            return response.json()['hasNextPage']

        return False

    curr_page = 1
    while _append_rawdata_if_has_next(curr_page):
        curr_page += 1

    return rawdata_ids


def get_rawdata(rawdata_str: str) -> RawData:
    """Gets the rawdata info.

    Args:
        rawdata (str): _description_
    """
    response = requests.get(os.path.join(DATACENTER_RAW_DATA_URL, rawdata_str),
                            timeout=TIMEOUT,
                            headers={"Content-Type": "application/json"}
                            )
    if response.status_code:
        rawdata = RawData()
        rawdata.rawdata_id = response.json()['_id']
        rawdata.data_key = response.json()['dataKey']
        rawdata.path = response.json()['path']

        if 'videos' in response.json():
            for video in response.json()['videos']:
                raw_video = RawVideo()
                if 'path' in video:
                    raw_video.video_path = video['path']
                if 'cameraKey' in video:
                    raw_video.camera_key = video['cameraKey']
                if 'frameFilePath' in video:
                    raw_video.frame_path = video['frameFilePath']
                if 'cameraConfig' in video and 'pos' in video['cameraConfig']:
                    raw_video.cam_pos = video['cameraConfig']['pos']
                    raw_video.calibration = video['cameraConfig']

                rawdata.videos.append(raw_video)

        if 'files' in response.json():
            for raw_file in response.json()['files']:
                if 'type' in raw_file and raw_file['type'] == 'lidar':
                    if 'error' not in raw_file['path']:
                        rawdata.lidar_path = raw_file['path']

    return rawdata


def create_dataset(name: str, description: str, rawdata_keys: List[str]):
    """Creates a new dataset.

    Args:
        name (str): name the dataset
        description (str): description of the dataset.
    """
    tags = ['MultiVision', 'Original Sampling']
    response = requests.post(DATACENTER_DATASET_URL,
                             json={'name': name,
                                   'description': description, 'tags': tags, 'rawDataKeys': rawdata_keys},
                             timeout=TIMEOUT,
                             headers={"Content-Type": "application/json"}
                             )
    if response.status_code != 201:
        raise Exception(f'Failed to create dataset: {response.text}')

    return response.json()['_id']


def post_frames(frames: List[FrameInfo]):
    """Posts the frames to the dataset.

    Args:
        frames (List[FrameInfo]): the list of frames to post.
        dataset_id (str): the dataset id to post the frames.
    """
    for frame in frames:
        json_data = {
            "name": frame.frame_name,
            "type:": "MultiVision",
            "tags": frame.tags,
            "frameIndex": frame.ffc_index,
            "rawDataId": frame.rawdata_id,
            "datasetId": frame.dataset_id,
            "channels": [
                {
                    "cameraKey": image.camera_key,
                    "image": {
                        "name": os.path.basename(image.image_path),
                        "path": image.image_path,
                        "thumbnailPath": image.webp_path,
                    }
                } for image in frame.channel_images
            ],
            "lidar": {
                "name": frame.lidar.name,
                "path": frame.lidar.path,
                "type": frame.lidar.type
            }
        }
        response = requests.post(os.path.join(DATACENTER_FRAME_URL),
                                 json=json_data,
                                 timeout=TIMEOUT,
                                 headers={"Content-Type": "application/json"}
                                 )

        if response.status_code != 201:
            logger.error(f'Failed to post frame: {response.text}')


def get_rawdata_keys_from_dataset(dataset_id: str):
    """Gets the rawdata keys from the dataset.

    Args:
        dataset_id (str): the dataset id to get the rawdata keys.
    """
    response = requests.get(os.path.join(DATACENTER_DATASET_URL, dataset_id),
                            timeout=TIMEOUT,
                            headers={"Content-Type": "application/json"}
                            )
    if response.status_code != 200:
        raise Exception(f'Failed to get dataset: {response.text}')

    return response.json()['rawDataKeys']


def update_status(rawdata_key: str, status: Status, output_list: List[str]):
    """Updates the dataset status.

    Args:
        dataset_id (str): the dataset id to update the status.
        status (str): the status to update.
    """
    rawdata_status_url = os.path.join(
        DATACENTER_RAW_DATA_URL, rawdata_key, 'job-status')
    response = requests.patch(rawdata_status_url,
                              json={'action': 'Sampling',
                                    'status': status.value,
                                    'outputPath': output_list
                                    },
                              timeout=TIMEOUT,
                              headers={"Content-Type": "application/json"}
                              )
    if response.status_code != 200:
        raise Exception(f'Failed to update dataset status: {response.text}')
