import os
import tarfile
from tqdm import tqdm

LIDAR_EXTENSION = '.pcd'
COMPRESSED_LIDAR_EXTENSION = '.tar'


def remove_unmatched_lidar_images(export_dir: str):
    """Remove the unmatched lidar images from the export directory."""
    lidar_dir = os.path.join(export_dir, 'lidar')
    image_dir = os.path.join(export_dir, 'images')
    if not os.path.isdir(lidar_dir) or not os.path.isdir(image_dir):
        return
    image_names_without_extension = [os.path.splitext(os.path.basename(image))[0] for image in os.listdir(image_dir)]
    for lidar_file in os.listdir(lidar_dir):
        if not lidar_file.endswith(LIDAR_EXTENSION):
            os.remove(os.path.join(lidar_dir, lidar_file))
            continue
        lidar_file_without_extension = os.path.splitext(lidar_file)[0]
        if lidar_file_without_extension not in image_names_without_extension:
            os.remove(os.path.join(lidar_dir, lidar_file))


def compress_lidar_and_images(export_dir: str):
    """Compress the lidar and images to tar format.

    Pairs lidar .pcd with matching image by basename. Prefer .png then .jpg/.jpeg.
    """
    lidar_dir = os.path.join(export_dir, 'lidar')
    image_dir = os.path.join(export_dir, 'images')
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    tar_path = os.path.join(export_dir, os.path.basename(export_dir) + COMPRESSED_LIDAR_EXTENSION)
    with tarfile.open(tar_path, 'w') as tar:
        for lidar_file in tqdm(sorted(os.listdir(lidar_dir)), desc='Compressing to tar...'):
            if not lidar_file.endswith(LIDAR_EXTENSION):
                continue
            compressed_lidar_path = os.path.join('lidar', lidar_file)
            base_name = os.path.splitext(lidar_file)[0]
            # Prefer png, fallback to jpg/jpeg
            candidate_images = [
                base_name + '.png',
                base_name + '.jpg',
                base_name + '.jpeg'
            ]
            chosen_image = None
            for candidate in candidate_images:
                if os.path.exists(os.path.join(image_dir, candidate)):
                    chosen_image = candidate
                    break

            if chosen_image:
                compressed_image_path = os.path.join('images', chosen_image)
                tar.add(os.path.join(image_dir, chosen_image), arcname=compressed_image_path)
                tar.add(os.path.join(lidar_dir, lidar_file), arcname=compressed_lidar_path)
    remove_unmatched_lidar_images(export_dir)


