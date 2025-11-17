""" This script contains Calibration class and related functions for consistent and integrate usage of the Calibration objects.

Should be contain the Calibration related codes here.
"""
import re
import numpy as np
from util.data_converter_util import DataConverter

SUPPORTED_DISTORTION_MODELS = ['standard', 'fisheye', 'generic8']

def quaternion_to_rotation_matrix(quaternion: np.ndarray, translation: np.ndarray = None):
    """ Covert a quaternion into a full three-dimensional rotation matrix.

    Args:
        quaternion: A 4 element array representing the quaternion (w, x, y, z)
        translation: A 3 element array representing the translation.

    Return:
        rotation_translation_matrix: [3, 4] rotation and translation matrix. [R|t]
    """
    norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1.0, atol=1e-12):
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError(
                "Normalize quaternioning with norm=0 would lead to division by zero.")
        quaternion = quaternion / norm

    # Extract the values from Q
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rotation_matrix = np.array([[r00, r01, r02],
                                [r10, r11, r12],
                                [r20, r21, r22]])

    rotation_translation_matrix = np.zeros([4, 4], dtype=np.float64)
    rotation_translation_matrix[-1, -1] = 1.
    rotation_translation_matrix[:3, :3] = rotation_matrix
    if translation is not None:
        rotation_translation_matrix[:3, 3] = translation

    return rotation_translation_matrix


class Calibration(object):
    """3D object which contains information about 2D / 3D coordinates and the other variables.

    Attributes:
        camera_type: Camera type. 'standard(rectlinear)' or 'fisheye'.
        camera_to_image_projection_matrix: Camera to image projection 3 x 4 matrix [intrinsic, translation]. For Kitti data, translation used.
        vcs_to_ccs_matrix: VCS to CCS projection 3 x 4 matrix [rotation, translation].
        camera_distortion_coefficient: Camera distortion coefficient 1 x 6 matrix [K1, K2, K3, K4, P1, P2].
    """

    def __init__(self, json_path=None, sv_txt_path=None, dict=None):
        self.width = None
        self.height = None
        self.camera_type = None
        self.camera_intrinsic_matrix = np.zeros([3, 4], dtype=np.float64)
        self.image_rectification_matrix = np.zeros([3, 3], dtype=np.float64)

        # Distortion coefficient: [k1, k2, k3, k4, p1/k5, p2/k6, k7, k8]
        # standard/fisheye: uses first 6 [k1, k2, k3, k4, p1, p2]
        # generic8: uses all 8 [k1, k2, k3, k4, k5, k6, k7, k8]
        self.camera_distortion_coefficient = np.zeros([8], dtype=np.float64)
        self.distortion_model = self.camera_type  # 'standard', 'fisheye', 'generic8'
        
        if self.camera_type not in SUPPORTED_DISTORTION_MODELS:
            self.distortion_model = 'standard'
        else:
            self.distortion_model = self.camera_type
    
        
        self.vcs_to_ccs_matrix = np.zeros([4, 4], dtype=np.float64)
        self.converter = DataConverter()
        if json_path is not None:
            self.load_from_json(json_path)
        if sv_txt_path is not None:
            self.load_from_sv_txt(sv_txt_path)
        if dict is not None:
            self.load_from_dict(dict)

    def save_to_json(self, save_path: str):
        extrinsic = self.vcs_to_ccs_matrix.flatten().tolist()
        intrinsic = self.camera_intrinsic_matrix.flatten().tolist()
        coefficient = self.camera_distortion_coefficient.flatten().tolist()
        assert len(intrinsic) in [9, 12]
        assert len(extrinsic) == 16
        assert len(coefficient) in [4, 6, 8]
        self.converter.dict = dict(
            type=self.camera_type,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            coefficient=coefficient
        )
        self.converter.dict_to_json().save_json(save_path)
        return self

    def load_from_json(self, json_path: str):
        self.load_from_dict(self.converter.read_json(
            json_path).json_to_dict().dict)
        return self

    def load_from_dict(self, params_dict: dict):
        params = params_dict
        assert len(params['intrinsic']) in [9, 12]
        assert len(params['extrinsic']) == 16
        # [K1, K2, P1, P2] or [K1, K2, K3, K4, P1, P2] or [K1-K8]
        assert len(params['coefficient']) in [4, 6, 8]
        self.camera_type = params['type'].lower(
        ) if 'type' in params else 'standard'

        # distortion model
        if 'distortion_model' in params:
            self.distortion_model = params['distortion_model'].lower()
        else:
            self.distortion_model = self.camera_type

        intrinsic = params['intrinsic']
        extrinsic = params['extrinsic']
        coefficient = params['coefficient']

        if self.camera_type == 'standard' or self.camera_type == 'rectlinear':
            if len(coefficient) == 4:
                # [K1, K2, P1, P2] -> [K1, K2, 0, 0, P1, P2, 0, 0]
                coefficient = [coefficient[0], coefficient[1],
                               0, 0, coefficient[2], coefficient[3], 0, 0]
            elif len(coefficient) == 6:
                # [K1, K2, K3, K4, P1, P2] -> [K1, K2, K3, K4, P1, P2, 0, 0]
                coefficient = list(coefficient) + [0, 0]
            elif len(coefficient) == 8:
                coefficient = list(coefficient)
        elif self.camera_type == 'fisheye':
            if len(coefficient) == 4:
                # [K1, K2, K3, K4] -> [K1, K2, K3, K4, 0, 0, 0, 0]
                coefficient = [coefficient[0], coefficient[1],
                               coefficient[2], coefficient[3], 0, 0, 0, 0]
            elif len(coefficient) == 6:
                # [K1, K2, K3, K4, P1, P2] -> [K1, K2, K3, K4, P1, P2, 0, 0]
                coefficient = list(coefficient) + [0, 0]
            elif len(coefficient) == 8:
                coefficient = list(coefficient)
        elif self.camera_type == 'generic8':
            if len(coefficient) == 8:
                # [K1-K8] -> already in correct format
                coefficient = list(coefficient)
            else:
                raise ValueError(f'generic8 requires 8 coefficients, got {len(coefficient)}')

        coefficient = np.asarray(coefficient, dtype=np.float64)

        if len(intrinsic) == 9:
            intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape([3, 3])
            intrinsic = np.concatenate([intrinsic, np.zeros([3, 1])], axis=1)
        else:
            intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape([3, 4])

        extrinsic = np.asarray(extrinsic, dtype=np.float64).reshape([4, 4])

        self.camera_intrinsic_matrix = intrinsic
        self.vcs_to_ccs_matrix = extrinsic
        self.camera_distortion_coefficient = coefficient

        if 'width' in params:
            self.width = params['width']
        if 'height' in params:
            self.height = params['height']
        return self

    def load_from_sv_txt(self, param_path: str):
        quad_w = None
        quad_x = None
        quad_y = None
        quad_z = None
        trans_x = None
        trans_y = None
        trans_z = None
        k5_val = 0.0
        k6_val = 0.0
        k7_val = 0.0
        k8_val = 0.0
        
        
        with open(param_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace(' ', '')
                finds = re.findall(';.*', line)
                for f in finds:
                    line = line.replace(f, '')
                finds = re.findall('#.*', line)
                for f in finds:
                    line = line.replace(f, '')
                if line == '':
                    continue
                key, value = line.split('=')
                value = value.replace('.F', '')
                if key == 'CAM_FX':
                    self.camera_intrinsic_matrix[0, 0] = float(value)
                elif key == 'CAM_FY':
                    self.camera_intrinsic_matrix[1, 1] = float(value)
                elif key == 'CAM_CX':
                    self.camera_intrinsic_matrix[0, 2] = float(value)
                elif key == 'CAM_CY':
                    self.camera_intrinsic_matrix[1, 2] = float(value)
                elif key == 'CAM_K1':
                    self.camera_distortion_coefficient[0] = float(value)
                elif key == 'CAM_K2':
                    self.camera_distortion_coefficient[1] = float(value)
                elif key == 'CAM_K3':
                    self.camera_distortion_coefficient[2] = float(value)
                elif key == 'CAM_K4':
                    self.camera_distortion_coefficient[3] = float(value)
                elif key == 'CAM_K5':
                    k5_val = float(value)
                elif key == 'CAM_K6':
                    k6_val = float(value)
                elif key == 'CAM_K7':
                    k7_val = float(value)
                elif key == 'CAM_K8':
                    k8_val = float(value)
               
                elif key == 'CAM_P1':
                    self.camera_distortion_coefficient[4] = float(value)
                elif key == 'CAM_P2':
                    self.camera_distortion_coefficient[5] = float(value)
                elif key in ['CAM_DISTORTION', 'CAM_TYPE']:
                    self.camera_type = value.lower()
                    if self.camera_type == 'rectlinear':
                        self.camera_type = 'standard'
                    elif self.camera_type == 'fisheye':
                        self.camera_type = 'fisheye'
                elif key == 'PIXEL_WIDTH':
                    self.width = int(value)
                elif key == 'PIXEL_HEIGHT':
                    self.height = int(value)
                elif key == 'QUAD_W':
                    quad_w = float(value)
                elif key == 'QUAD_X':
                    quad_x = float(value)
                elif key == 'QUAD_Y':
                    quad_y = float(value)
                elif key == 'QUAD_Z':
                    quad_z = float(value)
                elif key == 'TRANS_X':
                    trans_x = float(value)
                elif key == 'TRANS_Y':
                    trans_y = float(value)
                elif key == 'TRANS_Z':
                    trans_z = float(value)
                elif key == 'LIDAR_PITCH':
                    lidar_pitch = float(value)
                elif key == 'LIDAR_ROLL':
                    lidar_roll = float(value)
                elif key == 'LIDAR_YAW':
                    lidar_yaw = float(value)
                elif key == 'LIDAR_TX':
                    lidar_tx = float(value)
                elif key == 'LIDAR_TY':
                    lidar_ty = float(value)
                elif key == 'LIDAR_TZ':
                    lidar_tz = float(value)
                else:
                    print(f'Cannot recognize: {key}={value}')

        self.camera_intrinsic_matrix[2, 2] = 1
        
        if any([k5_val != 0, k6_val != 0, k7_val != 0, k8_val != 0]):
            # If K5-K8 exist, treat as Generic8 model and map K5-K8 to indices [4..7]
            self.camera_type = 'generic8'
            self.distortion_model = 'generic8'
            # Overwrite P1/P2 with zeros since Generic8 has no tangential terms
            self.camera_distortion_coefficient[4] = k5_val
            self.camera_distortion_coefficient[5] = k6_val
            self.camera_distortion_coefficient[6] = k7_val
            self.camera_distortion_coefficient[7] = k8_val
        
        if quad_w is not None and trans_x is not None:
            rot_mat = quaternion_to_rotation_matrix(
                quaternion=np.array(
                    [quad_w, quad_x, quad_y, quad_z], dtype=np.float64),
                translation=np.array(
                    [trans_x, trans_y, trans_z], dtype=np.float64)
            )
            self.vcs_to_ccs_matrix = rot_mat
        else:
            self.vcs_to_ccs_matrix = np.array([
                0, -1, 0, 0,
                0, 0, -1, 0,
                1, 0, 0, 0,
                0, 0, 0, 1]).reshape(4, 4)
    
                

    def read_kitti_raw_calib(self, raw_calib_cam_to_cam_file_path: str, raw_calib_velo_to_cam_file_path: str):
        _calib = dict()
        with open(raw_calib_cam_to_cam_file_path, 'r') as f:
            lines = f.readlines()
            calib_time_line = lines[0]
            corner_dist_line = lines[1]

            for line in lines[2:]:
                chars = line.replace(':', '').strip().split(' ')
                _calib[chars[0]] = [float(x) for x in chars[1:]]

        with open(raw_calib_velo_to_cam_file_path, 'r') as f:
            lines = f.readlines()
            calib_time_line = lines[0]

            for line in lines[1:]:
                chars = line.replace(':', '').strip().split(' ')
                _calib[chars[0]] = [float(x) for x in chars[1:]]

        self.camera_type = 'standard'

        self.camera_intrinsic_matrix = np.array(
            _calib['P_rect_02'], dtype=np.float64).reshape([3, 4])
        self.image_rectification_matrix = np.zeros([4, 4])
        self.image_rectification_matrix[0:3, 0:3] = np.array(
            _calib['R_rect_00'], dtype=np.float64).reshape([3, 3])
        self.image_rectification_matrix[3, 3] = 1

        self.vcs_to_ccs_matrix = np.zeros([4, 4])
        self.vcs_to_ccs_matrix[0:3, :] = np.concatenate([np.array(_calib['R'], dtype=np.float64).reshape(
            [3, 3]), np.array(_calib['T'], dtype=np.float64).reshape([3, 1])], axis=1)
        self.vcs_to_ccs_matrix[3, 3] = 1
        self.camera_distortion_coefficient = np.zeros([6], dtype=np.float64)
        return self

    def is_distorted(self):
        """ Check if the camera is distorted or not
        Returns:
            bool: True if the camera is distorted, False otherwise
        """
        return not np.allclose(self.camera_distortion_coefficient, 0.0, atol=1e-5)
    @property
    def cv_dist_coef(self):
        if self.camera_type == 'standard':
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
        elif self.camera_type == 'fisheye':
            return np.array([self.k1, self.k2, self.k3, self.k4])
        

    @property
    def hfov(self):
        return np.rad2deg(2 * np.arctan(self.width / (2 * self.fx)))

    @property
    def vfov(self):
        return np.rad2deg(2 * np.arctan(self.height / (2 * self.fy)))

    @property
    def fx(self):
        return self.camera_intrinsic_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_intrinsic_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_intrinsic_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_intrinsic_matrix[1, 2]

    @property
    def k1(self):
        return self.camera_distortion_coefficient[0]

    @property
    def k2(self):
        return self.camera_distortion_coefficient[1]

    @property
    def k3(self):
        return self.camera_distortion_coefficient[2]

    @property
    def k4(self):
        return self.camera_distortion_coefficient[3]
    
  
    @property
    def p1(self):
        return self.camera_distortion_coefficient[4]

    @property
    def p2(self):
        return self.camera_distortion_coefficient[5]
