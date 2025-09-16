
import numpy as np
from typing import List
from copy import deepcopy
from common.calibration import Calibration
import matplotlib.pyplot as plt
from util.coordinate_util import vcs_to_ics_points, vcs_to_ccs_points
import cv2


def color_by_coordinate_vector(xyz: np.ndarray, maximum_distance:float=70):
    """
    get color by coordinate vector
    Args:
        xyz: numpy array of shape (N, 3)
        maximum_distance: maximum distance to color
    Returns:
        colors: numpy array of shape (N, 3)
    """
    distance = np.clip(np.sqrt(np.power(xyz[:, 0], 2.) + np.power(xyz[:, 1], 2.)), 0, maximum_distance)
    distance_normalized = distance / maximum_distance

    cmap = plt.cm.get_cmap("rainbow", 256)
    colors = np.round(cmap(distance_normalized)[:, :3] * 255).astype(np.int32)

    return colors


def mapped_from_hsl():
    bgr_list = []
    for hue in range(120, -1, -1): # blue nearest
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        bgr_list.append(bgr)
    return bgr_list

distance_color_map = mapped_from_hsl()

def distance_to_color(xyzi: np.ndarray):
    """
    get color by coordinate vector
    Args:
        xyzi: numpy array of shape (N, 4). (x, y, z, intensity)
    Returns:
        colors: numpy array of shape (N, 3)
    """
    if len(xyzi) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    distances = np.linalg.norm(xyzi[:,0:3], axis=1)
    distance_normalized = distances / distances.max()
    colors = np.zeros((len(distance_normalized), 3), dtype=np.uint8)
    distanc_idxes = np.clip((distance_normalized * len(distance_color_map)).astype(np.int32), 0, len(distance_color_map)-1)
    for i, idx in enumerate(distanc_idxes):
        colors[i] = distance_color_map[idx]
    return colors

def intensity_to_color(xyzi: np.ndarray):
    """
    get color by intensity
    Args:
        xyzi: numpy array of shape (N, 4). (x, y, z, intensity)
    Returns:
        colors: numpy array of shape (N, 3)
    """
    if len(xyzi) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    intensity = xyzi[:, 3]
    intensity_normalized = intensity / intensity.max()
    colors = np.zeros((len(intensity_normalized), 3), dtype=np.uint8)
    distanc_idxes = np.clip((intensity_normalized * len(distance_color_map)).astype(np.int32), 0, len(distance_color_map)-1)
    for i, idx in enumerate(distanc_idxes):
        colors[i] = distance_color_map[idx]
    return colors

def draw_projected_lidar_point_cloud_to_camera_image(
        point_cloud_in_vcs:np.ndarray,
        image:np.ndarray,
        calib:Calibration,
        image_drawing_scale:float=1.0,
        translation:List[int]=[0, 0],
        base_on_intensity:bool=False,
        point_radius:int=1,
        alpha:float=1.0,
        ):
    """
    Draw Projected Lidar Point Cloud to Camera Image.
    Args:
        point_cloud_in_vcs: (N, 3) or (N, 4) array
        image: (H, W, 3) array
        calib: Calibration object
        image_drawing_scale: (w_scale, h_scale) tuple
        draw_points_on_image: draw points on image
        draw_mouse_near_points: draw mouse near points
        lidar_near_mouse_points_mask: numpy array of shape (N,). True if point is near mouse.
        lidar_dist_from_mouse: numpy array of shape (N,). distance from mouse.
        dist_threshold: float
        base_on_intensity: if true, pixel color is based on intensity. if not, pixel color is based on distance.
    Return:
        image: (H, W, 3) array
    """

    h, w = image.shape[:2]

    rescaled_calib = deepcopy(calib)
    rescaled_calib.camera_intrinsic_matrix[0, 0] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[0, 2] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[0, 3] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[1, 1] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[1, 2] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[1, 3] *= image_drawing_scale
    rescaled_calib.camera_intrinsic_matrix[2, 2] = 1.

    projected_points = np.round(vcs_to_ics_points(point_cloud_in_vcs[:, :3], rescaled_calib)[:, :2]).astype(np.int32)
    projected_points[:, 0] += translation[0]
    projected_points[:, 1] += translation[1]
    point_cloud_in_ccs = vcs_to_ccs_points(point_cloud_in_vcs[:, :3], rescaled_calib)
    in_image_point_indices = (
        (projected_points[:, 0] >= 1) *
        (projected_points[:, 0] < w - 1) *
        (projected_points[:, 1] >= 1) *
        (projected_points[:, 1] < h - 1) *
        (point_cloud_in_ccs[:, 2] >= 0)
    )
    selected_projected_points = projected_points[in_image_point_indices]
    selected_point_cloud = point_cloud_in_vcs[in_image_point_indices]
    
    #colors = color_by_coordinate_vector(selected_point_cloud)[:, ::-1]
    if base_on_intensity:
        colors = intensity_to_color(selected_point_cloud)
    else:
        colors = distance_to_color(selected_point_cloud)

    # draw points with adjustable size and optional alpha blending
    overlay = image.copy()
    pr = max(1, int(point_radius))
    for (x, y), (b, g, r) in zip(selected_projected_points, colors):
        cv2.circle(overlay, (int(x), int(y)), pr, (int(b), int(g), int(r)), thickness=-1, lineType=cv2.LINE_AA)

    if alpha >= 1.0:
        image[:, :, :3] = overlay[:, :, :3]
        return image
    else:
        a = max(0.0, min(1.0, float(alpha)))
        blended = cv2.addWeighted(overlay, a, image, 1.0 - a, 0.0)
        image[:, :, :3] = blended[:, :, :3]
        return image
