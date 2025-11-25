import os
import re
import json
import tarfile
import shutil
import cv2
from tqdm import tqdm
from loguru import logger

from parser.cam_info_parser import get_calib_from_cam_info_dict
from util.aclm_util import get_aclm_sampling_required_files
from util.cam_util import (
    get_timestamps_from_frame_file,
    SampleFrame,
)
from util.lidar_util import copy_sampled_lidar, read_pcd_file
from tools.lidar_to_cam_matcher import (
    calculate_motor_angle_range,
    find_matching_cam_frames_from_circular_lidar,
)
from dto.sample_frame import write_sample_frames_to_csv, SAMPLE_CSV_NAME
from util.compress_util import compress_lidar_and_images
from util.raw_converter import AclmRawConverter, RawOutputFormat
from util.draw_util import draw_projected_lidar_point_cloud_to_camera_image


def _extract_raw_images_from_tar_by_indices(image_tar_path: str, frame_indices: list[int], result_dir: str):
    """Extract .raw images from tar file based on frame indices."""
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
                # ACLM uses .raw extension instead of .jpg/.png
                if not lower_name.endswith('.raw'):
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
        logger.error(f"Failed to extract raw images from tar: {image_tar_path}, error: {e}")
    return extracted_image_names


def _rename_lidar_using_sample_frames(generated_images: list[str], sample_frames: list[SampleFrame], lidar_dir: str, lidar_extension: str = '.pcd'):
    """Rename lidar files to match image frame indices."""
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


def run_aclm_pipeline(rawdata_set: list[str], export: str, compress: bool, remove: bool, stride: int,
                      overlay_every: int = 0, overlay_intensity: bool = False, overlay_point_radius: int = 2,
                      overlay_alpha: float = 1.0, skip_head: int = 0, skip_tail: int = 0,
                      convert_raw_to_jpg: bool = False, raw_output_format: str = 'gray', raw_dgain: float = 1.5):
    """
    Run ACLM sampling pipeline.

    ACLM uses raw image format (.raw) instead of processed images.
    Optionally convert raw to jpg for visualization.

    Args:
        rawdata_set: List of raw data directories
        export: Export directory
        compress: Compress to tar
        remove: Remove uncompressed files after compression
        stride: Sample every Nth LiDAR frame
        overlay_every: Generate overlay every N pairs (only if convert_raw_to_jpg=True)
        overlay_intensity: Color by intensity instead of distance
        overlay_point_radius: Point radius for overlay
        overlay_alpha: Alpha blending for overlay
        skip_head: Skip N frames from beginning
        skip_tail: Skip N frames from end
        convert_raw_to_jpg: Convert .raw files to .jpg (default: False)
        raw_output_format: Output format for conversion - 'gray' or 'rgb' (default: 'gray')
        raw_dgain: Digital gain for RGB conversion (default: 1.5)
    """
    os.makedirs(export, exist_ok=True)
    os.makedirs(os.path.join(export, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(export, 'raw'), exist_ok=True)

    if convert_raw_to_jpg:
        os.makedirs(os.path.join(export, 'images'), exist_ok=True)

    for raw in tqdm(rawdata_set, desc='Processing ACLM'):
        try:
            image_tar_file, frame_txt_file, cam_info_file, lidar_tar_file = get_aclm_sampling_required_files(raw)

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

                # Extract .raw files to raw/ folder
                generated_raw_files = _extract_raw_images_from_tar_by_indices(
                    image_tar_path=image_tar_file,
                    frame_indices=[sf.frame_id for sf in sample_frames],
                    result_dir=os.path.join(export, 'raw'),
                )

                if generated_raw_files:
                    _rename_lidar_using_sample_frames(
                        generated_images=generated_raw_files,
                        sample_frames=sample_frames,
                        lidar_dir=os.path.join(export, 'lidar'),
                    )

                    # Optional: Convert .raw to .jpg
                    if convert_raw_to_jpg:
                        try:
                            output_fmt = RawOutputFormat.RGB if raw_output_format.lower() == 'rgb' else RawOutputFormat.GRAY
                            converter = AclmRawConverter(output_format=output_fmt, dgain=raw_dgain)

                            for raw_filename in generated_raw_files:
                                raw_path = os.path.join(export, 'raw', os.path.basename(raw_filename))
                                if not os.path.exists(raw_path):
                                    continue

                                # Read raw data
                                with open(raw_path, 'rb') as f:
                                    raw_data = f.read()

                                # Convert to JPG
                                jpg_data = converter.convert_raw_to_jpg_bytes(raw_data)

                                # Save to images/
                                base_name = os.path.splitext(os.path.basename(raw_filename))[0]
                                jpg_path = os.path.join(export, 'images', f"{base_name}.jpg")
                                with open(jpg_path, 'wb') as f:
                                    f.write(jpg_data)

                        except ImportError as e:
                            logger.error(f"Failed to convert raw to jpg: {e}")
                            logger.error("Install dependencies: pip install scipy opencv-python")
                        except Exception as e:
                            logger.error(f"Raw to jpg conversion error: {e}")

                    # Optional: Generate overlay (only if conversion is enabled)
                    if convert_raw_to_jpg and overlay_every and overlay_every > 0:
                        overlays_dir = os.path.join(export, 'overlays')
                        os.makedirs(overlays_dir, exist_ok=True)

                        # Get converted JPG files
                        jpg_files = sorted([f for f in os.listdir(os.path.join(export, 'images')) if f.endswith('.jpg')])

                        for idx, jpg_name in enumerate(jpg_files):
                            if idx % overlay_every != 0:
                                continue

                            base = os.path.splitext(jpg_name)[0]
                            img_path = os.path.join(export, 'images', jpg_name)
                            pcd_path = os.path.join(export, 'lidar', base + '.pcd')

                            if not os.path.exists(img_path) or not os.path.exists(pcd_path):
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
            logger.error(f"Failed to handle ACLM raw: {raw}, error: {e}")
            continue

    if compress:
        print("Compress images and lidars...")
        compress_lidar_and_images(export)
        if remove:
            if os.path.exists(os.path.join(export, 'images')):
                shutil.rmtree(os.path.join(export, 'images'))
            if os.path.exists(os.path.join(export, 'raw')):
                shutil.rmtree(os.path.join(export, 'raw'))
            shutil.rmtree(os.path.join(export, 'lidar'))
