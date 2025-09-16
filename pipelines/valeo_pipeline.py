import os
import re
import json
import tarfile
import shutil
import cv2
from tqdm import tqdm
from loguru import logger

from parser.cam_info_parser import get_calib_from_cam_info_dict
from util.valeo_util import get_valeo_sampling_required_files
from util.cam_util import (
    get_timestamps_from_frame_file,
    SampleFrame,
)
from util.lidar_util import copy_sampled_lidar
from tools.lidar_to_cam_matcher import (
    calculate_motor_angle_range,
    find_matching_cam_frames_from_circular_lidar,
)
from dto.sample_frame import write_sample_frames_to_csv, SAMPLE_CSV_NAME
from util.compress_util import compress_lidar_and_images
from util.lidar_util import read_pcd_file
from util.draw_util import draw_projected_lidar_point_cloud_to_camera_image


def _extract_images_from_tar_by_indices(image_tar_path: str, frame_indices: list[int], result_dir: str):
    extracted_image_names = []
    if not image_tar_path or not os.path.exists(image_tar_path):
        return extracted_image_names
    os.makedirs(result_dir, exist_ok=True)
    index_set = set(frame_indices)
    try:
        with tarfile.open(image_tar_path, 'r') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                lower_name = member.name.lower()
                if not (lower_name.endswith('.png') or lower_name.endswith('.jpg') or lower_name.endswith('.jpeg')):
                    continue
                base_name = os.path.basename(member.name)
                name_wo_ext, _ = os.path.splitext(base_name)
                match = re.search(r'(\d+)$', name_wo_ext)
                if not match:
                    continue
                frame_idx = int(match.group(1))
                if frame_idx in index_set:
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    out_path = os.path.join(result_dir, base_name)
                    with open(out_path, 'wb') as out_f:
                        out_f.write(fileobj.read())
                    extracted_image_names.append(base_name)
    except Exception as e:
        logger.error(f"Failed to extract images from tar: {image_tar_path}, error: {e}")
    return extracted_image_names


def _rename_lidar_using_sample_frames(generated_images: list[str], sample_frames: list[SampleFrame], lidar_dir: str, lidar_extension: str = '.pcd'):
    if not generated_images or not sample_frames:
        return
    frame_id_to_lidar_ts = {sf.frame_id: sf.lidar_timestamp for sf in sample_frames}
    for image in generated_images:
        base_name = os.path.splitext(os.path.basename(image))[0]
        match = re.search(r'(\d+)$', base_name)
        if not match:
            continue
        frame_idx = int(match.group(1))
        lidar_ts = frame_id_to_lidar_ts.get(frame_idx)
        if lidar_ts is None:
            continue
        lidar_src = os.path.join(lidar_dir, f"{lidar_ts}{lidar_extension}")
        if not os.path.exists(lidar_src):
            continue
        lidar_dst = os.path.join(lidar_dir, f"{base_name}{lidar_extension}")
        try:
            os.rename(lidar_src, lidar_dst)
        except Exception as e:
            logger.error(f"Failed to rename lidar {lidar_src} -> {lidar_dst}: {e}")


def run_valeo_pipeline(rawdata_set: list[str], export: str, compress: bool, remove: bool, stride: int, overlay_every: int = 0, overlay_intensity: bool = False, overlay_point_radius: int = 2, overlay_alpha: float = 1.0):
    os.makedirs(export, exist_ok=True)
    os.makedirs(os.path.join(export, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(export, 'images'), exist_ok=True)

    for raw in tqdm(rawdata_set, desc='Progress'):
        try:
            image_tar_file, frame_txt_file, cam_info_file, lidar_tar_file = get_valeo_sampling_required_files(raw)

            if not cam_info_file:
                logger.error(f"Failed to find the cam info file for the raw: {raw}")
                continue
            if not frame_txt_file:
                logger.error(f"Failed to find the frame txt file for the raw: {raw}")
                continue
            if not lidar_tar_file:
                logger.error(f"Failed to find the lidar file for the raw: {raw}")
                continue

            with open(cam_info_file, 'r', encoding='utf-8') as file:
                cam_info_dict = json.load(file)
                calib_info = get_calib_from_cam_info_dict(cam_info_dict)
                angle_start, angle_end = calculate_motor_angle_range(calib_info)
                frame_dict = get_timestamps_from_frame_file(frame_txt_file)

                lidar_frames = copy_sampled_lidar(lidar_tar_file, sampling_ratio=stride, copy_dir=export)
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

                generated_images = _extract_images_from_tar_by_indices(
                    image_tar_path=image_tar_file,
                    frame_indices=[sf.frame_id for sf in sample_frames],
                    result_dir=os.path.join(export, 'images'),
                )

                if generated_images:
                    _rename_lidar_using_sample_frames(
                        generated_images=generated_images,
                        sample_frames=sample_frames,
                        lidar_dir=os.path.join(export, 'lidar'),
                    )
                    # Optional: create overlay webp every Nth pair
                    if overlay_every and overlay_every > 0:
                        overlays_dir = os.path.join(export, 'overlays')
                        os.makedirs(overlays_dir, exist_ok=True)
                        for idx, img_name in enumerate(sorted(generated_images)):
                            if idx % overlay_every != 0:
                                continue
                            base = os.path.splitext(os.path.basename(img_name))[0]
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

            write_sample_frames_to_csv(sample_frames, os.path.join(export, os.path.basename(raw) + "_" + SAMPLE_CSV_NAME))
        except Exception as e:
            logger.error(f"Failed to handle VALEO raw: {raw}, error: {e}")
            continue

    if compress:
        print("Compress images and lidars...")
        compress_lidar_and_images(export)
        if remove:
            shutil.rmtree(os.path.join(export, 'images'))
            shutil.rmtree(os.path.join(export, 'lidar'))


