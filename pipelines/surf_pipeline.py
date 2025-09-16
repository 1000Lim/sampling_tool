import os
import json
import shutil
import cv2
from tqdm import tqdm
from loguru import logger

from parser.cam_info_parser import get_calib_from_cam_info_dict
from util.rawdata_util import (
    get_cam_info_file,
    get_sampling_lidar,
    get_sampling_txt,
    get_sampling_video,
)
from util.cam_util import (
    get_timestamps_from_frame_file,
    sampling_target_frames_from_video,
    SampleFrame,
)
from util.lidar_util import copy_sampled_lidar, rename_sampled_lidar
from tools.lidar_to_cam_matcher import (
    calculate_motor_angle_range,
    find_matching_cam_frames_from_circular_lidar,
)
from dto.sample_frame import write_sample_frames_to_csv, SAMPLE_CSV_NAME
from util.compress_util import compress_lidar_and_images
from util.lidar_util import read_pcd_file
from util.draw_util import draw_projected_lidar_point_cloud_to_camera_image


def run_surf_pipeline(rawdata_set: list[str], export: str, compress: bool, remove: bool, cam_info: str, cam_num: int, stride: int, overlay_every: int = 0, overlay_intensity: bool = False, overlay_point_radius: int = 2, overlay_alpha: float = 1.0, skip_head: int = 0, skip_tail: int = 0):
    os.makedirs(export, exist_ok=True)
    os.makedirs(os.path.join(export, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(export, 'images'), exist_ok=True)

    for raw in tqdm(rawdata_set, desc='Progress'):
        sample_frames = []
        try:
            cam_info_path = cam_info
            if not cam_info_path:
                cam_info_path = get_cam_info_file(raw, cam_num)
                if not cam_info_path:
                    logger.error(f"Failed to find the cam info file for the raw: {raw}")
                    continue

            video_file = get_sampling_video(raw, cam_num)
            frame_file = get_sampling_txt(video_file)
            lidar_file = get_sampling_lidar(raw)

            if not video_file:
                logger.error(f"Failed to find the video file for the raw: {raw}")
                continue

            if not frame_file:
                logger.error(f"Failed to find the frame txt file for the raw: {raw}")
                continue

            if not lidar_file:
                logger.error(f"Failed to find the lidar file for the raw: {raw}")
                continue

            with open(cam_info_path, 'r', encoding='utf-8') as file:
                cam_info_dict = json.load(file)
                calib_info = get_calib_from_cam_info_dict(cam_info_dict)
                angle_start, angle_end = calculate_motor_angle_range(calib_info)
                frame_dict = get_timestamps_from_frame_file(frame_file)
                lidar_frames = copy_sampled_lidar(lidar_file, sampling_ratio=stride, copy_dir=export)
                if skip_head > 0 or skip_tail > 0:
                    start_idx = min(skip_head, len(lidar_frames))
                    end_idx = len(lidar_frames) - max(skip_tail, 0)
                    end_idx = max(start_idx, end_idx)
                    lidar_frames = lidar_frames[start_idx:end_idx]
                lidar_matched_frames = find_matching_cam_frames_from_circular_lidar(
                    angle_start, angle_end, lidar_frames, frame_dict, export_interface_file=os.path.join(export, 'mapping.csv')
                )

                sample_frames = [
                    SampleFrame(
                        frame_id=lidar_matched_frame.camera_index,
                        frame_timestamp=lidar_matched_frame.camera_fsync_end,
                        lidar_timestamp=lidar_matched_frame.lidar_timestamp,
                    )
                    for lidar_matched_frame in lidar_matched_frames
                ]

                generated_images = sampling_target_frames_from_video(
                    video_path=video_file,
                    sampling_frames=sample_frames,
                    result_dir=os.path.join(export, 'images'),
                    ignore_frames=[0, 1, 2],
                )

                rename_sampled_lidar(generated_images, os.path.join(export, 'lidar'))
                # Optional: write overlays for every Nth pair
                if overlay_every and overlay_every > 0:
                    overlays_dir = os.path.join(export, 'overlays')
                    os.makedirs(overlays_dir, exist_ok=True)
                    for idx, img_name in enumerate(sorted(generated_images)):
                        if idx % overlay_every != 0:
                            continue
                        base = os.path.splitext(os.path.basename(img_name))[0]
                        # choose image extension
                        img_path = None
                        for ext in ('.png', '.jpg', '.jpeg'):
                            candidate = os.path.join(export, 'images', base + ext)
                            if os.path.exists(candidate):
                                img_path = candidate
                                break
                        if not img_path:
                            continue
                        pcd_path = os.path.join(export, 'lidar', base + '.pcd')
                        if not os.path.exists(pcd_path):
                            continue
                        try:
                            image = cv2.imread(img_path)
                            points_xyzi = read_pcd_file(pcd_path)
                            overlay = draw_projected_lidar_point_cloud_to_camera_image(
                                point_cloud_in_vcs=points_xyzi,
                                image=image,
                                calib=calib_info,
                                base_on_intensity=overlay_intensity,
                                point_radius=overlay_point_radius,
                                alpha=overlay_alpha,
                            )
                            out_path = os.path.join(overlays_dir, base + '.webp')
                            cv2.imwrite(out_path, overlay)
                        except Exception as e:
                            logger.error(f"Failed to create overlay for {base}: {e}")
                write_sample_frames_to_csv(
                    sample_frames, os.path.join(export, os.path.basename(raw) + "_" + SAMPLE_CSV_NAME)
                )
        except Exception as e:
            logger.error(f"Failed to sample the raw: {raw}, error: {e}")
            continue

    if compress:
        print("Compress images and lidars...")
        compress_lidar_and_images(export)
        if remove:
            shutil.rmtree(os.path.join(export, 'images'))
            shutil.rmtree(os.path.join(export, 'lidar'))


