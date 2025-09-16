"""This module parsing the calib to Calibration class.
"""
from dataclasses import dataclass, field
import numpy as np
from common.calibration import Calibration, quaternion_to_rotation_matrix


@dataclass
class LidarToCamera:
    """The Camera View Range class.
    """
    quad_w: float = None
    quad_x: float = None
    quad_y: float = None
    quad_z: float = None
    trans_x: float = None
    trans_y: float = None
    trans_z = float = None


@dataclass
class CalibrationParams:
    """The Camera View Range class.
    """
    distortion: str = None
    cx: float = None
    cy: float = None
    fx: float = None
    fy: float = None
    k1: float = None
    k2: float = None
    k3: float = None
    k4: float = None
    p1: float = None
    p2: float = None
    lidar_to_camera: LidarToCamera = field(default_factory=LidarToCamera)


def get_calib_from_cam_info_json_dict(item: dict) -> Calibration:
    """Parse the calib to Calibration class.

    Args:
        item (dict): A dictionary containing the calibration information.

    Returns:
        Calibration: An instance of the Calibration class.
    """
    calib = Calibration()
    calib.width = item['width']
    calib.height = item['height']
    cam_calibration = item['calibration']
    calib.camera_type = cam_calibration['distortion'].lower()

    # intrinsic matrix
    intrinsic = calib.camera_intrinsic_matrix
    intrinsic[0, 0] = cam_calibration['fx']
    intrinsic[1, 1] = cam_calibration['fy']
    intrinsic[0, 2] = cam_calibration['cx']
    intrinsic[1, 2] = cam_calibration['cy']
    intrinsic[2, 2] = 1
    distortion = calib.camera_distortion_coefficient
    distortion[0] = cam_calibration['k1']
    distortion[1] = cam_calibration['k2']
    distortion[2] = cam_calibration['k3']
    distortion[3] = cam_calibration['k4']
    distortion[4] = cam_calibration['p1']
    distortion[5] = cam_calibration['p2']
    # lidar calibration
    lidar_param = item['lidar_parameter']

    rot_mat = quaternion_to_rotation_matrix(
        quaternion=np.array([
            lidar_param['quad_w'],
            lidar_param['quad_x'],
            lidar_param['quad_y'],
            lidar_param['quad_z']
        ], dtype=np.float64),
        translation=np.array([
            lidar_param['trans_x'],
            lidar_param['trans_y'],
            lidar_param['trans_z']
        ], dtype=np.float64)
    )
    calib.vcs_to_ccs_matrix = rot_mat
    return calib


def parse_calib_param_cam_info(items: dict):
    """Parse the calibration parameters from the camera info dictionary.

    Args:
        items (dict): A dictionary containing the calibration information.

    Returns:
        CalibrationParams: CameraCalibration dataclass
    """
    camera_calib = CalibrationParams()
    cam_calibration = items['calibration']
    camera_calib.distortion = cam_calibration['distortion']
    camera_calib.cx = cam_calibration['cx']
    camera_calib.cy = cam_calibration['cy']
    camera_calib.fx = cam_calibration['fx']
    camera_calib.fy = cam_calibration['fy']
    camera_calib.k1 = cam_calibration['k1']
    camera_calib.k2 = cam_calibration['k2']
    camera_calib.k3 = cam_calibration['k3']
    camera_calib.k4 = cam_calibration['k4']
    camera_calib.p1 = cam_calibration['p1']
    camera_calib.p2 = cam_calibration['p2']

    if 'lidar_parameter' in items:
        lidar_param = items['lidar_parameter']
        camera_calib.lidar_to_camera.quad_w = lidar_param['quad_w']
        camera_calib.lidar_to_camera.quad_x = lidar_param['quad_x']
        camera_calib.lidar_to_camera.quad_y = lidar_param['quad_y']
        camera_calib.lidar_to_camera.quad_z = lidar_param['quad_z']
        camera_calib.lidar_to_camera.trans_x = lidar_param['trans_x']
        camera_calib.lidar_to_camera.trans_y = lidar_param['trans_y']
        camera_calib.lidar_to_camera.trans_z = lidar_param['trans_z']

    return camera_calib


def get_calib_from_camera_dc_camera_config(camera_config: dict) -> Calibration:
    """Parse the calib to Calibration class.

    Args:
        camera_config (dict): A dictionary containing the calibration information.

    Returns:
        Calibration: An instance of the Calibration class.
    """
    if not all(key in camera_config for key in ['width', 'height', 'cameraCalibration', 'lidarCalibration']):
        return None

    calib = Calibration()
    calib.width = camera_config['width']
    calib.height = camera_config['height']
    camera_calib = camera_config['cameraCalibration']

    calib.camera_type = camera_calib['distortion'].lower()
    # intrinsic matrix
    intrinsic = calib.camera_intrinsic_matrix
    intrinsic[0, 0] = camera_calib['fx']
    intrinsic[1, 1] = camera_calib['fy']
    intrinsic[0, 2] = camera_calib['cx']
    intrinsic[1, 2] = camera_calib['cy']
    intrinsic[2, 2] = 1
    distortion = calib.camera_distortion_coefficient
    distortion[0] = camera_calib['k1']
    distortion[1] = camera_calib['k2']
    distortion[2] = camera_calib['k3']
    distortion[3] = camera_calib['k4']
    distortion[4] = camera_calib['p1']
    distortion[5] = camera_calib['p2']
    # lidar calibration
    lidar_param = camera_config['lidarCalibration']

    rot_mat = quaternion_to_rotation_matrix(
        quaternion=np.array([
            lidar_param['quad']['w'],
            lidar_param['quad']['x'],
            lidar_param['quad']['y'],
            lidar_param['quad']['z']
        ], dtype=np.float64),
        translation=np.array([
            lidar_param['trans']['x'],
            lidar_param['trans']['y'],
            lidar_param['trans']['z']
        ], dtype=np.float64)
    )
    calib.vcs_to_ccs_matrix = rot_mat
    return calib
