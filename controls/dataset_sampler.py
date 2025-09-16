"""This functions manage the dataset sampling process."""


import os
import asyncio
from functools import partial
from typing import List
from util import lidar_util, cam_util
from rest import data_center_rest as dc_api
from dto.frame_info import FrameInfo, LidarInfo
from consts import MOUNTED_DATA_CENTER_PATH, LIDAR_EXTENSION, POS_DICT, Status
from tools import lidar_to_cam_matcher
from parser import calib_parser
from loguru import logger


async def run_dataset_sampling(rawdata_keys: List[str], dataset_id: str,
                               sampling_ratio: int = 1, mv_only: bool = True):
    """Functions for run dataset sampling.

    Args:
        dataset_id (str): dataset id.
        sampling_ratio (int, optional): lidar per image.
    """
    loop = asyncio.get_running_loop()
    sampling_frames: List[FrameInfo] = []
    # sampling front channel.

    for data in rawdata_keys:
        rawdata = dc_api.get_rawdata(data)
        await loop.run_in_executor(None, partial(dc_api.update_status, data, Status.PROCESSING, []))
        # sampling lidar.
        rawdata_basename = rawdata.path
        if rawdata_basename.startswith('/'):
            rawdata_basename = rawdata_basename[1:]

        rawdata_dir = os.path.join(
            MOUNTED_DATA_CENTER_PATH, rawdata_basename)

        if not os.path.isfile(os.path.join(rawdata_dir, os.path.basename(rawdata.lidar_path))):
            raise FileNotFoundError(
                f'lidar file: {os.path.join(rawdata_dir, os.path.basename(rawdata.lidar_path))} not found.')

        lidar_timestamps = await loop.run_in_executor(None, partial(
            lidar_util.extract_sampled_lidar,
            extracted_tar=os.path.join(
                rawdata_dir, os.path.basename(rawdata.lidar_path)),
            sampling_ratio=sampling_ratio,
            copy_dir=rawdata_dir)
        )
        # sampling camera.
        ffc_video = None
        cam_images_dict = dict()

        for video in rawdata.videos:
            if video.camera_key.lower() == '2160p_h120_ffc':
                ffc_video = video
                break

        if not ffc_video:
            raise FileNotFoundError(
                f'ffc video not found in rawdata: {rawdata.data_key}')

        ffc_image_dict = await loop.run_in_executor(None, partial(
            cam_util.get_timestamps_from_frame_file,
            os.path.join(rawdata_dir, os.path.basename(
                ffc_video.frame_path))
        ))

        # parse calibration from camera config.
        calib = await loop.run_in_executor(None, partial(
            calib_parser.get_calib_from_camera_dc_camera_config,
            ffc_video.calibration))

        if not calib:
            raise ValueError(
                f'calibration can not be parsed(cameraCalibration, lidarCalibration) can not found in rawdata: {rawdata.data_key}')

        angle_start, angle_end = await loop.run_in_executor(None, partial(
            lidar_to_cam_matcher.calculate_motor_angle_range,
            calib)
        )
        logger.info(f'mapping angle is around {angle_start} to {angle_end}.')
        mapping_file_path = os.path.join(rawdata_dir, os.path.splitext(
            os.path.basename(ffc_video.video_path))[0] + '_mapping.txt')
        lidar_matched_ffc_images = await loop.run_in_executor(None, partial(
            lidar_to_cam_matcher.find_matching_cam_frames_from_circular_lidar,
            motor_begin_angle=angle_start,
            motor_end_angle=angle_end,
            lidar_timestamps=lidar_timestamps,
            camera_timestamps=ffc_image_dict,
            export_interface_file=mapping_file_path)
        )
        ffc_images = await loop.run_in_executor(None, partial(
            cam_util.get_ffc_images_with_matched_frames,
            matched_frames=lidar_matched_ffc_images,
            camera_key=ffc_video.camera_key,
        ))

        # original way to sampling ffc images.
        # sampling ffc images.
        # ffc_images = await loop.run_in_executor(None, partial(
        #     cam_util.get_sample_images_with_lidar_timestamps_with_time_offset,
        #     lidar_timestamps,  # timestamps
        #     25,  # offset
        #     os.path.join(rawdata_dir, os.path.basename(
        #         ffc_video.frame_path)),
        #     ffc_video.camera_key)
        # )

        for image in ffc_images:
            frame_name = rawdata.data_key + '_' + str(image.lidar_timestamp)
            # if use lidar rename
            # lidar=LidarInfo(
            #     name=frame_name + LIDAR_EXTENSION,
            #     path=os.path.join(
            #         rawdata_basename, 'lidar', frame_name + LIDAR_EXTENSION)
            # )

            # if not use lidar rename
            lidar = LidarInfo(
                name=str(image.lidar_timestamp) + LIDAR_EXTENSION,
                path=os.path.join(
                    rawdata_basename, 'lidar', str(
                        image.lidar_timestamp) + LIDAR_EXTENSION
                )
            )

            frame_info = FrameInfo(
                frame_name=frame_name,
                ffc_index=image.image_index,
                ffc_timestamp=image.image_timestamp,
                lidar_timestamp=image.lidar_timestamp,
                dataset_id=dataset_id,
                rawdata_id=rawdata.rawdata_id,
                channel_images=[image],
                lidar=lidar
            )
            sampling_frames.append(frame_info)
        cam_images_dict[ffc_video.camera_key] = ffc_images

        other_videos = [video for video in rawdata.videos if video.cam_pos.lower(
        ) != 'ffc' and video.cam_pos]

        for other_video in other_videos:
            # find other cam images based on ffc timestamps.
            if mv_only:
                if other_video.camera_key.lower() not in POS_DICT:
                    continue

            other_video_path = os.path.join(
                rawdata_dir, os.path.basename(other_video.frame_path))
            if not os.path.isfile(other_video_path):
                raise FileNotFoundError(
                    f'other video file: {other_video_path} not found.')

            other_cam_images = await loop.run_in_executor(None, partial(
                cam_util.get_sample_images_with_ffc_images,
                ffc_images,  # timestamps
                other_video_path),
                other_video.camera_key
            )
            cam_images_dict[other_video.camera_key] = other_cam_images

            for frame in sampling_frames:
                for image in other_cam_images:
                    if image.lidar_timestamp == frame.lidar_timestamp:
                        frame.channel_images.append(image)
                        break

        sampling_tasks = []
        output_list = [os.path.join(rawdata.path, 'lidar')]
        for video in rawdata.videos:
            if mv_only:
                mapping_pos = POS_DICT.get(
                    video.camera_key.lower(), None)
            else:
                mapping_pos = POS_DICT.get(
                    video.camera_key.lower(), video.camera_key.lower())

            if video.cam_pos and video.camera_key in cam_images_dict and mapping_pos:
                output_list.append(os.path.join(
                    rawdata.path, 'images', mapping_pos))
                sampling_task = loop.run_in_executor(None,
                                                     cam_util.sampling_target_frames_from_video,
                                                     os.path.join(rawdata_dir, os.path.basename(
                                                         video.video_path)),
                                                     rawdata.data_key,
                                                     cam_images_dict[video.camera_key],
                                                     os.path.join(
                                                         rawdata_dir, 'images', mapping_pos)
                                                     )
                sampling_tasks.append(sampling_task)
        await asyncio.gather(*sampling_tasks)
        # await loop.run_in_executor(None, partial(lidar_util.rename_pcd_files, os.path.join(rawdata_dir, 'lidar'), rawdata.data_key))
        await loop.run_in_executor(None, partial(dc_api.post_frames, sampling_frames))
        await loop.run_in_executor(None, partial(dc_api.update_status, data, Status.COMPLETED, output_list))
