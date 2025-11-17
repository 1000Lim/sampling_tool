"""this module supports parsing camera information from cam_info.json file.
"""
import numpy as np
from common.calibration import Calibration
from util.coordinate_util import quaternion_to_rotation_matrix


def _parse_legacy_cam_info(item: dict) -> Calibration:
    """Parse legacy SURF-style cam_info with keys 'calibration' and 'lidar_parameter'."""
    calib = Calibration()

    calib.width = item['width']
    calib.height = item['height']
    cam_calibration = item['calibration']

    # Set camera type
    distortion_type = cam_calibration.get('distortion', '').lower()
    if distortion_type:
        calib.camera_type = distortion_type
        calib.distortion_model = distortion_type

    intrinsic = calib.camera_intrinsic_matrix
    intrinsic[0, 0] = float(cam_calibration['fx'])
    intrinsic[1, 1] = float(cam_calibration['fy'])
    intrinsic[0, 2] = float(cam_calibration['cx'])
    intrinsic[1, 2] = float(cam_calibration['cy'])
    intrinsic[2, 2] = 1.0

    distortion = calib.camera_distortion_coefficient
    distortion[0] = float(cam_calibration.get('k1', 0.0))
    distortion[1] = float(cam_calibration.get('k2', 0.0))
    distortion[2] = float(cam_calibration.get('k3', 0.0))
    distortion[3] = float(cam_calibration.get('k4', 0.0))

    if distortion_type == 'generic8':
        # generic8 uses k1-k8
        distortion[4] = float(cam_calibration.get('k5', 0.0))
        distortion[5] = float(cam_calibration.get('k6', 0.0))
        distortion[6] = float(cam_calibration.get('k7', 0.0))
        distortion[7] = float(cam_calibration.get('k8', 0.0))
    else:
        distortion[4] = float(cam_calibration.get('p1', 0.0))
        distortion[5] = float(cam_calibration.get('p2', 0.0))

    lidar_param = item['lidar_parameter']
    rot_mat = quaternion_to_rotation_matrix(
        quaternion=np.array([
            float(lidar_param['quad_w']),
            float(lidar_param['quad_x']),
            float(lidar_param['quad_y']),
            float(lidar_param['quad_z'])
        ], dtype=np.float64),
        translation=np.array([
            float(lidar_param['trans_x']),
            float(lidar_param['trans_y']),
            float(lidar_param['trans_z'])
        ], dtype=np.float64)
    )
    calib.vcs_to_ccs_matrix = rot_mat
    return calib


def get_calib_from_valeo_cam_info_dict(item: dict) -> Calibration:
    """Parse Valeo-format cam_info JSON into Calibration.

    Expected schema:
      - width, height at top-level
      - parameter.camera_parameter: CAM_FX, CAM_FY, CAM_CX, CAM_CY, CAM_DISTORTION,
        CAM_K1..K4, CAM_P1, CAM_P2
      - parameter.lidar_parameter: QUAD_W/X/Y/Z and TRANS_X/Y/Z (preferred)
        Fallback keys may include LIDAR_PITCH/ROLL/YAW and LIDAR_TX/TY/TZ (not used here)
    """
    calib = Calibration()
    calib.width = item['width']
    calib.height = item['height']

    params = item['parameter']
    cam_params = params['camera_parameter']
    lidar_params = params['lidar_parameter']

    # Set camera type
    distortion_type = cam_params.get('CAM_DISTORTION', '').lower()
    if distortion_type:
        calib.camera_type = distortion_type

    # intrinsic
    intrinsic = calib.camera_intrinsic_matrix
    intrinsic[0, 0] = float(cam_params['CAM_FX'])
    intrinsic[1, 1] = float(cam_params['CAM_FY'])
    intrinsic[0, 2] = float(cam_params['CAM_CX'])
    intrinsic[1, 2] = float(cam_params['CAM_CY'])
    intrinsic[2, 2] = 1.0

    # distortion
    distortion = calib.camera_distortion_coefficient
    distortion[0] = float(cam_params.get('CAM_K1', 0.0))
    distortion[1] = float(cam_params.get('CAM_K2', 0.0))
    distortion[2] = float(cam_params.get('CAM_K3', 0.0))
    distortion[3] = float(cam_params.get('CAM_K4', 0.0))

    if distortion_type == 'generic8':
        # generic8 uses k1-k8
        distortion[4] = float(cam_params.get('CAM_K5', 0.0))
        distortion[5] = float(cam_params.get('CAM_K6', 0.0))
        distortion[6] = float(cam_params.get('CAM_K7', 0.0))
        distortion[7] = float(cam_params.get('CAM_K8', 0.0))
    else:
        distortion[4] = float(cam_params.get('CAM_P1', 0.0))
        distortion[5] = float(cam_params.get('CAM_P2', 0.0))

    # lidar extrinsic via quaternion + translation
    qw = float(lidar_params['QUAD_W'])
    qx = float(lidar_params['QUAD_X'])
    qy = float(lidar_params['QUAD_Y'])
    qz = float(lidar_params['QUAD_Z'])
    tx = float(lidar_params['TRANS_X'])
    ty = float(lidar_params['TRANS_Y'])
    tz = float(lidar_params['TRANS_Z'])

    rot_mat = quaternion_to_rotation_matrix(
        quaternion=np.array([qw, qx, qy, qz], dtype=np.float64),
        translation=np.array([tx, ty, tz], dtype=np.float64)
    )
    calib.vcs_to_ccs_matrix = rot_mat
    return calib


def get_calib_from_cam_info_dict(item: dict) -> Calibration:
    """Auto-detect cam_info schema and return Calibration for both legacy and Valeo formats.

    - Legacy format: top-level 'calibration' and 'lidar_parameter'
    - Valeo format: top-level 'parameter' with 'camera_parameter' and 'lidar_parameter'
    """
    # Valeo format detection
    if isinstance(item.get('parameter'), dict) and \
       'camera_parameter' in item['parameter'] and 'lidar_parameter' in item['parameter']:
        return get_calib_from_valeo_cam_info_dict(item)

    # Legacy format detection
    if 'calibration' in item and 'lidar_parameter' in item:
        return _parse_legacy_cam_info(item)

    # Unknown format
    raise KeyError('Unsupported cam_info schema: missing expected keys')