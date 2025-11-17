""" Utility functions abount coordinate system transformation, etc.

Abbreviations:
    - ics: Image Coordinate System
    - vcs: Vehicle Coordinate System (same as Lidar Coordinate System)
    - ccs: Camera Coordinate System

Please refer to below confluence page for coordinate system explanation.
https://stradvision.atlassian.net/wiki/spaces/CDG/pages/47291044882/DCT-3DDBEV+Data
"""
import numpy as np
from common.calibration import Calibration
import cv2
from typing import Union


EPS = 1e-8

def make_radian_in_range(radian:Union[float, np.ndarray]):
    """ Make radian in range [-pi, +pi].
    Args:
        radian (float): Radian value.
    Returns:
        float: Radian value in range [-pi, +pi].
    """
    if isinstance(radian, np.ndarray):
        new_radian = radian.copy()
        new_radian = new_radian % (2 * np.pi)
        new_radian[new_radian > np.pi] = new_radian[new_radian > np.pi] - 2 * np.pi
    elif isinstance(radian, float) or isinstance(radian, np.float32) or isinstance(radian, np.float64):
        new_radian = radian
        new_radian = new_radian % (2 * np.pi)
        if new_radian > np.pi:
            new_radian = new_radian - 2 * np.pi
    else:
        raise TypeError(f"radian type {type(radian)} is not supported.")
    return new_radian


# For KITTI dataset ===========================================================
def rotation_y_to_alpha(rotation_y:float, ccs_x:float, ccs_z:float):
    """Convert rotation to alpha.

    Args:
        rotation_y (float): Rotation of object around Y-axis.
        ccs_x (float): X-coordinate of object in CCS.
        ccs_z (float): Z-coordinate of object in CCS.
    Returns:
        float: Alpha of object.
    """
    arctan2 = np.arctan2
    alpha = rotation_y - arctan2(ccs_x, ccs_z)
    alpha = make_radian_in_range(alpha)

    return alpha


def alpha_to_rotation_y(alpha:float, ccs_x:float, ccs_z:float):
    """ Convert alpha to rotation.
    Args:
        alpha (float): Alpha of object.
        ccs_x (float): X-coordinate of object in CCS.
        ccs_z (float): Z-coordinate of object in CCS.
    Returns:
        float: Rotation of object around Y-axis.
    """
    arctan2 = np.arctan2
    rotation_y = alpha + arctan2(ccs_x, ccs_z)
    rotation_y = make_radian_in_range(rotation_y)

    return rotation_y


def yaw_to_rotation_y(yaw:float):
    """Convert VCS yaw angle to CCS rotation_y.

    Args:
        yaw (float): VCS yaw angle.

    Returns:
        float: Rotation_y angle.
    """
    rot_y = -np.pi / 2 - yaw

    rot_y = make_radian_in_range(rot_y)

    return rot_y


def rotation_y_to_yaw(rot_y:float):
    """Convert CCS rotation_y to VCS yaw angle.

    Args:
        rot_y (float): CCS rotation y angle.

    Returns:
        float: Rotation_y angle.
    """
    yaw = -np.pi / 2 - rot_y

    yaw = make_radian_in_range(yaw)

    return yaw


def alpha_to_yaw(alpha:float, ccs_x:float, ccs_z:float):
    """ Convert alpha to yaw.
    Args:
        alpha (float): Relative angle of object in image.
        ccs_x (float): X-coordinate of object in CCS.
        ccs_z (float): Z-coordinate of object in CCS.
    Return:
        float: VCS yaw angle.
    """
    return rotation_y_to_yaw(alpha_to_rotation_y(alpha, ccs_x, ccs_z))


def yaw_to_alpha(yaw:float, ccs_x:float, ccs_z:float):
    """ Convert yaw to alpha.
    Args:
        yaw (float): VCS yaw angle.
        ccs_x (float): X-coordinate of object in CCS.
        ccs_z (float): Z-coordinate of object in CCS.
    Return:
        float: Relative angle of object in image.
    """
    return rotation_y_to_alpha(yaw_to_rotation_y(yaw), ccs_x, ccs_z)
# =============================================================================


def cart_to_homo(x:np.ndarray):
    """Convert transformation matrix in Cartesian coordinates to
    homogeneous format.

    Args:
        mat (np.ndarray): Transformation matrix in Cartesian.
            The input matrix shape is 3x3 or 3x4.

    Returns:
        np.ndarray: Transformation matrix in homogeneous format.
            The matrix shape is 4x4.
    """
    ret = np.eye(4)
    if x.shape == (3, 3):
        ret[:3, :3] = x
    elif x.shape == (3, 4):
        ret[:3, :] = x
    else:
        raise ValueError(x.shape)
    return ret


def cartesian_to_homogeneous(arr:np.ndarray):
    """
    Args:
        arr: [..., [A0, ..., An]] matrix
    Return:
        [..., [A0, ..., An, 1]] matrix
    """
    temp = np.ones(arr.shape[:-1] + (arr.shape[-1] + 1,), dtype=arr.dtype)
    temp[..., :-1] = arr
    return temp


def ccs_to_vcs_points(ccs_points:np.ndarray, calib:Calibration=None):
    """ Transform the CCS points array to the VCS points array.
    Args:
        ccs_points: [..., [x, y, z]] xyz points array at CCS.
    Return:
        vcs_points: [..., [x, y, z]] xyz points array at VCS.
    """
    assert ccs_points.shape[-1] == 3
    if calib is None:  # Just convert coordinate system
        vcs_points = np.zeros_like(ccs_points)
        vcs_points[..., 0] = ccs_points[..., 2]
        vcs_points[..., 1] = -ccs_points[..., 0]
        vcs_points[..., 2] = -ccs_points[..., 1]
    else:
        vcs_to_ccs_matrix_inv = np.linalg.inv(calib.vcs_to_ccs_matrix).astype(np.float32)
        vcs_points = (cartesian_to_homogeneous(ccs_points) @ vcs_to_ccs_matrix_inv.T)[..., :3]

    return vcs_points


def vcs_to_ccs_points(vcs_points:np.ndarray, calib:Calibration=None):
    """ Transform the VCS points array to the CCS points array.
    Args:
        vcs_points: [..., [x, y, z]] xyz points array at VCS.
    Return:
        ccs_points: [..., [x, y, z]] xyz points array at CCS.
    """
    # assert vcs_points.shape[-1] == 3
    if calib is None:  # Just convert coordinate system
        ccs_points = np.zeros_like(vcs_points)
        ccs_points[..., 0] = -vcs_points[..., 1]
        ccs_points[..., 1] = -vcs_points[..., 2]
        ccs_points[..., 2] = vcs_points[..., 0]
    else:
        vcs_to_ccs_matrix = calib.vcs_to_ccs_matrix
        ccs_points = (cartesian_to_homogeneous(vcs_points) @ vcs_to_ccs_matrix.T)[..., :3]

    return ccs_points


def ccs_to_ics_points(ccs_points:np.ndarray, calib:Calibration, only_in_image:bool=False, new_intrinsic_calib:Calibration=None):
    """ Transform the CCS points array to the ICS points array.
    Args:
        ccs_points: [..., [x, y, z]] xyz points array at CCS.
        calib: Calibration object.
        only_in_image: If True, only return the points in the image.
    Return:
        ics_points: [..., [x, y, z]] xyz points array at ICS.
    """
    assert ccs_points.shape[-1] == 3

    intrinsic = calib.camera_intrinsic_matrix
    abs = np.abs
    cat = np.concatenate

    ics_points = (cartesian_to_homogeneous(ccs_points) @ intrinsic.T)[..., :3]
    ics_xy = ics_points[..., :2]
    ics_depth = ics_points[..., 2:3]
    ics_depth[abs(ics_depth) < EPS] = EPS
    ics_xy = ics_xy / ics_depth  # To avoid inplace operation
    ics_points = cat([ics_xy, ics_depth], -1)
    distorted_ics_points = apply_distortion_at_ics_points(ics_points, calib, new_intrinsic_calib)

    if only_in_image:
        mask = (distorted_ics_points[..., 0] >= 0) & (distorted_ics_points[..., 0] < calib.cx * 2) & \
               (distorted_ics_points[..., 1] >= 0) & (distorted_ics_points[..., 1] < calib.cy * 2)
        distorted_ics_points = distorted_ics_points[mask]

    return distorted_ics_points


def ics_to_ccs_points(ics_points:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None):
    """ Transform the ICS points array to the CCS points array.
    Args:
        ics_points: [..., [x, y, z]] xyz points array at ICS.
        calib: Calibration object.
    Return:
        ccs_points: [..., [x, y, z]] xyz points array at CCS.
    """
    assert ics_points.shape[-1] == 3

    intrinsic_inv = np.linalg.inv(cart_to_homo(calib.camera_intrinsic_matrix)).astype(np.float32)
    undistorted_ics_points = apply_undistortion_at_ics_points(ics_points, calib, new_intrinsic_calib)
    undistorted_ics_points[..., :2] *= undistorted_ics_points[..., 2:3].copy()
    ccs_points = (cartesian_to_homogeneous(undistorted_ics_points) @ intrinsic_inv.T)[..., :3]

    return ccs_points


def apply_distortion_at_ics_points(undistorted_ics_points:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None):
    """ Transform the VCS points array to the ICS points array.
    Args:
        undistorted_ics_points: Undistorted [N, 3] xyz points array at ICS.
        calib: Calibration object.
    Return:
        distorted_points: Distorted [N, 3] xyz points array at ICS.
    """

    sqrt = np.sqrt
    arctan = np.arctan
    _copy = np.copy
    if not calib.is_distorted():
        undistorted_ics_points = _copy(undistorted_ics_points)
        if new_intrinsic_calib is not None:
            undistorted_ics_points = ccs_to_ics_points(ics_to_ccs_points(undistorted_ics_points, calib), new_intrinsic_calib)
        return undistorted_ics_points

    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    u = (undistorted_ics_points[..., 0:1] - cx) / fx
    v = (undistorted_ics_points[..., 1:2] - cy) / fy
    z = undistorted_ics_points[..., 2:3]

    if calib.camera_type == 'standard':
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        k1, k2, k3, k4, p1, p2 = calib.camera_distortion_coefficient[:6]
        r2 = (u ** 2) + (v ** 2)
        radial_d = (1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)) / (1 + k4 * r2)
        u = radial_d * u + 2 * p1 * (u * v) + p2 * (r2 + 2 * (u ** 2))
        v = radial_d * v + p1 * (r2 + 2 * (v ** 2)) + 2 * p2 * (u * v)
    elif calib.camera_type == 'fisheye':
        # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
        k1, k2, k3, k4, p1, p2 = calib.camera_distortion_coefficient[:6]
        r = sqrt(u ** 2 + v ** 2)
        theta = arctan(r)
        theta_d = theta + (k1 * theta ** 3) + (k2 * theta ** 5) + (k3 * theta ** 7) + (k4 * theta ** 9)
        u = (theta_d / r) * u
        v = (theta_d / r) * v
    elif calib.camera_type == 'generic8':
        # Generic8 distortion model
        # The polynomial outputs distorted radius in PIXEL space, not normalized space
        # r_distort_pixels = k1*θ + k2*θ² + ... + k8*θ⁸
        k1, k2, k3, k4, k5, k6, k7, k8 = calib.camera_distortion_coefficient
        r_undistort = sqrt(u ** 2 + v ** 2)
        r_sq_plus_1 = u ** 2 + v ** 2 + 1
        r = sqrt(r_sq_plus_1)
        r_inv = 1.0 / r

        # Only apply distortion where r_inv <= 1
        valid_mask = np.abs(r_inv) <= 1.0

        # Calculate theta (angle from optical axis)
        theta_u = np.arccos(np.clip(r_inv, -1.0, 1.0))
        theta_u_2 = theta_u * theta_u
        theta_u_3 = theta_u * theta_u_2
        theta_u_4 = theta_u_2 * theta_u_2
        theta_u_5 = theta_u_2 * theta_u_3
        theta_u_6 = theta_u_3 * theta_u_3
        theta_u_7 = theta_u_2 * theta_u_5
        theta_u_8 = theta_u_4 * theta_u_4

        # r_distort is in PIXEL space
        r_distort_pixels = (k1 * theta_u + k2 * theta_u_2 + k3 * theta_u_3 + k4 * theta_u_4 +
                            k5 * theta_u_5 + k6 * theta_u_6 + k7 * theta_u_7 + k8 * theta_u_8)

        # Convert to normalized space by dividing by focal length
        # Use average of fx and fy for radial distortion
        f_avg = (fx + fy) / 2.0
        r_distort_normalized = r_distort_pixels / f_avg

        # Calculate distortion factor
        factor = np.where(r_undistort > 1e-8, r_distort_normalized / r_undistort, 1.0)
        factor = np.where(np.isfinite(factor), factor, 1.0)
        factor = np.where(valid_mask, factor, 1.0)

        u = factor * u
        v = factor * v
    else:
        raise ValueError(f'Cannot recognize type of camera = {calib.camera_type}')

    if new_intrinsic_calib is not None:
        fx = new_intrinsic_calib.fx
        fy = new_intrinsic_calib.fy
        cx = new_intrinsic_calib.cx
        cy = new_intrinsic_calib.cy
    x = u * fx + cx
    y = v * fy + cy

    return np.concatenate([x, y, z], axis=-1)


# https://kornia.readthedocs.io/en/0.5.7/_modules/kornia/geometry/calibration/undistort.html
def apply_undistortion_at_ics_points(ics_points:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None):
    r"""Compensate for lens distortion a set of 2D image points.

    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.

    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`

    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.
    """
    tan = np.tan
    mean = np.mean
    _copy = np.copy

    if not calib.is_distorted():
        ics_points = _copy(ics_points)
        if new_intrinsic_calib is not None:
            ics_points = ccs_to_ics_points(ics_to_ccs_points(ics_points, calib), new_intrinsic_calib)
        return ics_points

    # Extract distortion coefficients based on camera type
    if calib.camera_type == 'generic8':
        k1, k2, k3, k4, k5, k6, k7, k8 = calib.camera_distortion_coefficient
        p1, p2 = 0.0, 0.0  # Generic8 has no tangential distortion
    else:
        k1, k2, k3, k4, p1, p2 = calib.camera_distortion_coefficient[:6]

    # Convert 2D points from pixels to normalized camera coordinates
    cx = calib.cx  # princial point in x (Bx1)
    cy = calib.cy  # princial point in y (Bx1)
    fx = calib.fx  # focal in x (Bx1)
    fy = calib.fy  # focal in y (Bx1)

    z = ics_points[..., 2:3]

    do_opencv = np.all(calib.camera_intrinsic_matrix[..., 3] == 0)

    if do_opencv:
        intrinsic = calib.camera_intrinsic_matrix[:3, :3]
        kwargs = dict(P=intrinsic)
        if new_intrinsic_calib is not None:
            kwargs['P'] = new_intrinsic_calib.camera_intrinsic_matrix[:3, :3]

        if calib.camera_type == 'fisheye':
            if len(ics_points.shape) == 1:
                pts = ics_points[np.newaxis, np.newaxis, :2]
            else:
                pts = ics_points[..., np.newaxis, :2]
            kwargs['distorted'] = np.ascontiguousarray(pts)
            kwargs['K'] = intrinsic
            kwargs['D'] = calib.cv_dist_coef
            undistort_points = cv2.fisheye.undistortPoints(**kwargs)
        elif calib.camera_type == 'standard':
            kwargs['src'] = np.ascontiguousarray(ics_points[..., :2])
            kwargs['cameraMatrix'] = intrinsic
            kwargs['distCoeffs'] = calib.cv_dist_coef
            undistort_points = cv2.undistortPoints(**kwargs)
        elif calib.camera_type == 'generic8':
            # Generic8 undistortion not supported by OpenCV
            # For now, return points without undistortion
            # TODO: Implement iterative undistortion for Generic8
            import warnings
            warnings.warn('Generic8 undistortion is not yet implemented. Returning distorted points.')
            x = (ics_points[..., 0:1] - cx) / fx
            y = (ics_points[..., 1:2] - cy) / fy
            if new_intrinsic_calib is not None:
                x = x * new_intrinsic_calib.fx + new_intrinsic_calib.cx
                y = y * new_intrinsic_calib.fy + new_intrinsic_calib.cy
            else:
                x = x * fx + cx
                y = y * fy + cy
            return np.concatenate([x, y, z], axis=-1)
        else:
            raise ValueError(f'Cannot recognize type of camera = {calib.camera_type}')
        x = undistort_points[..., 0, 0:1]
        y = undistort_points[..., 0, 1:2]
        x = x.reshape(z.shape)
        y = y.reshape(z.shape)
    else:
        # This is equivalent to K^-1 [u,v,1]^T
        u = (ics_points[..., 0:1] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
        v = (ics_points[..., 1:2] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

        if calib.camera_type == 'standard':
            # Iteratively undistort points
            x0, y0 = u, v
            for _ in range(5):
                r2 = u ** 2 + v ** 2

                inv_rad_poly = (1 + k4 * r2) / (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3)
                deltaX = 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
                deltaY = p1 * (r2 + 2 * v * v) + 2 * p2 * u * v

                u = (x0 - deltaX) * inv_rad_poly
                v = (y0 - deltaY) * inv_rad_poly

        elif calib.camera_type == 'fisheye':  # Refer from Sehoon Park's code (DRVPSTODP-47)
            dist_u = u.copy()
            dist_v = v.copy()
            r = (dist_u ** 2 + dist_v ** 2) ** 0.5
            r_ = r.copy()
            for c in range(20):
                theta_d = r_ + k1 * (r_ ** 3) + k2 * (r_ ** 5) + k3 * (r_ ** 7) + k4 * (r_ ** 9)
                error_distance = theta_d - r
                if abs(mean(error_distance)) < 1e-6:
                    break
                theta_d_dt = 1 + 3 * k1 * (r_ ** 2) + 5 * k2 * (r_ ** 4) + 7 * k3 * (r_ ** 6) + 9 * k4 * (r_ ** 8)
                r_ = r_ - error_distance / theta_d_dt

            undistorted_distance = tan(r_)
            u = dist_u * undistorted_distance / r
            v = dist_v * undistorted_distance / r
        else:
            raise ValueError(f'Cannot recognize type of camera = {calib.camera_type}')

        # Covert points from normalized camera coordinates to pixel coordinates
        if new_intrinsic_calib is not None:
            fx = new_intrinsic_calib.fx
            fy = new_intrinsic_calib.fy
            cx = new_intrinsic_calib.cx
            cy = new_intrinsic_calib.cy
        x = fx * u + cx
        y = fy * v + cy

    return np.concatenate([x, y, z], -1)


def undistort_image(img:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None):
    """ Undistort image using camera calibration
    Args:
        img (np.ndarray): Image to undistort
        calib (Calibration): Camera calibration
        new_camera_intrinsic (np.ndarray): New camera intrinsic matrix
    Returns:
        np.ndarray: Undistorted image
    """
    if new_intrinsic_calib is None:
        new_camera_intrinsic = calib.camera_intrinsic_matrix[:3, :3]
    else:
        new_camera_intrinsic = new_intrinsic_calib.camera_intrinsic_matrix[:3, :3]

    if calib.camera_type == 'standard':
        new_image = cv2.undistort(
            src=img,
            cameraMatrix=calib.camera_intrinsic_matrix[:3, :3],
            distCoeffs=calib.cv_dist_coef,
            newCameraMatrix=new_camera_intrinsic
        )
    elif calib.camera_type == 'fisheye':
        new_image = cv2.fisheye.undistortImage(
            distorted=img,
            K=calib.camera_intrinsic_matrix[:3, :3],
            D=calib.cv_dist_coef,
            Knew=new_camera_intrinsic
        )
    else:
        raise ValueError(f'Cannot recognize type of camera = {calib.camera_type}')
    return new_image


def undistort_image2(img:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None, subsampling:int=1):
    h, w, c = img.shape

    xs = np.linspace(0, w - 1, w // subsampling)
    ys = np.linspace(0, h - 1, h // subsampling)
    dst_points = cartesian_to_homogeneous(np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2))
    src_target_points = apply_distortion_at_ics_points(dst_points, calib, new_intrinsic_calib)

    mask = (0 <= src_target_points[..., 0]) & (src_target_points[..., 0] < w) & \
        (0 <= src_target_points[..., 1]) & (src_target_points[..., 1] < h)

    points_in_src_image = src_target_points[mask][..., :2]
    points_in_dst_image = (dst_points[mask][..., :2] / subsampling).astype(np.int32)

    new_image = np.zeros_like(img)
    if subsampling != 1:
        sample_map = np.zeros([h // subsampling, w // subsampling, 2], dtype=np.float32)
        sample_mask = np.zeros([h // subsampling, w // subsampling, 1], dtype=np.float32)
        sample_map[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = points_in_src_image
        sample_mask[np.all(sample_map != 0, -1)] = 1
        sample_map = cv2.resize(sample_map, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        sample_mask = cv2.resize(sample_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample_map = sample_map * sample_mask[..., np.newaxis]

        mask = np.all(sample_map != 0, axis=-1)
        masked_map = sample_map[mask].astype(np.int32)

        new_image[mask] = img[masked_map[..., 1], masked_map[..., 0]]
    else:
        points_in_src_image = points_in_src_image.astype(np.int32)
        new_image[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = img[points_in_src_image[..., 1], points_in_src_image[..., 0]]
    return new_image


def distort_image(img:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None, subsampling:int=1):
    h, w, c = img.shape

    xs = np.linspace(0, w - 1, w // subsampling)
    ys = np.linspace(0, h - 1, h // subsampling)
    dst_points = cartesian_to_homogeneous(np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2))
    src_target_points = apply_undistortion_at_ics_points(dst_points, calib, new_intrinsic_calib)

    mask = (0 <= src_target_points[..., 0]) & (src_target_points[..., 0] < w) & \
        (0 <= src_target_points[..., 1]) & (src_target_points[..., 1] < h)

    points_in_src_image = src_target_points[mask][..., :2]
    points_in_dst_image = (dst_points[mask][..., :2] / subsampling).astype(np.int32)

    new_image = np.zeros_like(img)
    if subsampling != 1:
        sample_map = np.zeros([h // subsampling, w // subsampling, 2], dtype=np.float32)
        sample_mask = np.zeros([h // subsampling, w // subsampling, 1], dtype=np.float32)
        sample_map[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = points_in_src_image
        sample_mask[np.all(sample_map != 0, -1)] = 1
        sample_map = cv2.resize(sample_map, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        sample_mask = cv2.resize(sample_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample_map = sample_map * sample_mask[..., np.newaxis]

        mask = np.all(sample_map != 0, axis=-1)
        masked_map = sample_map[mask].astype(np.int32)

        new_image[mask] = img[masked_map[..., 1], masked_map[..., 0]]
    else:
        points_in_src_image = points_in_src_image.astype(np.int32)
        new_image[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = img[points_in_src_image[..., 1], points_in_src_image[..., 0]]
    return new_image


def warp_image(img:np.ndarray, src_calib:Calibration, target_calib:Calibration=None, new_shape:np.ndarray=None, subsampling:int=1):
    if new_shape is None:
        new_shape = img.shape
    h, w, c = new_shape
    _h, _w, _c = img.shape

    xs = np.linspace(0, w - 1, w // subsampling)
    ys = np.linspace(0, h - 1, h // subsampling)
    dst_points = cartesian_to_homogeneous(np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2))
    src_target_points = ccs_to_ics_points(ics_to_ccs_points(dst_points, target_calib), src_calib)

    mask = (0 <= src_target_points[..., 0]) & (src_target_points[..., 0] < _w) & \
        (0 <= src_target_points[..., 1]) & (src_target_points[..., 1] < _h)

    points_in_src_image = src_target_points[mask][..., :2]
    points_in_dst_image = (dst_points[mask][..., :2] / subsampling).astype(np.int32)

    new_image = np.zeros_like(img)
    if subsampling != 1:
        sample_map = np.zeros([h // subsampling, w // subsampling, 2], dtype=np.float32)
        sample_mask = np.zeros([h // subsampling, w // subsampling, 1], dtype=np.float32)
        sample_map[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = points_in_src_image
        sample_mask[np.all(sample_map != 0, -1)] = 1
        sample_map = cv2.resize(sample_map, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        sample_mask = cv2.resize(sample_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample_map = sample_map * sample_mask[..., np.newaxis]

        mask = np.all(sample_map != 0, axis=-1) * (0 <= sample_map[..., 1]) * (sample_map[..., 1] < h) * (0 <= sample_map[..., 0]) * (sample_map[..., 0] < w)
        masked_map = sample_map[mask].astype(np.int32)

        new_image[mask] = img[masked_map[..., 1], masked_map[..., 0]]
    else:
        points_in_src_image = points_in_src_image.astype(np.int32)
        new_image[points_in_dst_image[..., 1], points_in_dst_image[..., 0]] = img[points_in_src_image[..., 1], points_in_src_image[..., 0]]
    return new_image


def vcs_to_ics_points(vcs_points:np.ndarray, calib:Calibration, new_intrinsic_calib:Calibration=None):
    """ Transform the VCS points array to the ICS points array.
    Args:
        vcs_points: [..., [x, y, z]] xyz points array at VCS.
        calib: Calibration object.
    Return:
        ics_points: [..., [x, y, z]] xyz points array at ICS.
    """
    return ccs_to_ics_points(vcs_to_ccs_points(vcs_points, calib), calib, new_intrinsic_calib=new_intrinsic_calib)


def ics_to_vcs_points(ics_points:np.ndarray, calib:Calibration):
    """ Transform the ICS points array to the VCS points array.
    Args:
        vcs_points: [..., [x, y, z]] xyz points array at ICS.
        calib: Calibration object.
    Return:
        ics_points: [..., [x, y, z]] xyz points array at VCS.
    """
    return ccs_to_vcs_points(ics_to_ccs_points(ics_points, calib), calib)


def vcs_bbox3d_to_vcs_corners(bbox3d:np.ndarray, order='xyz'):
    """
    Args:
        bbox3d: [..., [x, y, z, w, h, l, roll, pitch, yaw]] at VCS
    """
    trans_matrix = euler_angle_to_rotation_matrix(bbox3d[..., -3:], bbox3d[..., :3], order=order)
    ones_like = np.ones_like
    concat = np.concatenate
    matmul = np.matmul
    swapaxes = np.swapaxes
    stack = np.stack

    half_size = bbox3d[..., 3:6] / 2.
    w, h, l = half_size[..., :1], half_size[..., 1:2], half_size[..., 2:3]  # half of witdh, height and length
    flb = concat([l, w, -h, ones_like(l)], -1)  # front left bottom
    frb = concat([l, -w, -h, ones_like(l)], -1)  # front right bottom
    frt = concat([l, -w, h, ones_like(l)], -1)  # front right top
    flt = concat([l, w, h, ones_like(l)], -1)  # front left top
    rlb = concat([-l, w, -h, ones_like(l)], -1)  # rear left bottom
    rrb = concat([-l, -w, -h, ones_like(l)], -1)  # rear right bottom
    rrt = concat([-l, -w, h, ones_like(l)], -1)  # rear right top
    rlt = concat([-l, w, h, ones_like(l)], -1)  # rear left top
    local_points = stack(
        [flb, frb, frt, flt, rlb, rrb, rrt, rlt],
        -2
    )

    world_points = matmul(local_points, swapaxes(trans_matrix, -1, -2))[..., :3]  # 8 x 3 matrix
    return world_points


def ccs_bbox3d_to_ccs_corners(bbox3d:np.ndarray, order='xyz'):
    """
    Args:
        bbox3d: [x, y, z, w, h, l, roll, pitch, yaw] at CCS
    """
    trans_matrix = euler_angle_to_rotation_matrix(bbox3d[..., -3:], bbox3d[..., :3], order=order)

    asarray = np.asarray
    dtype = np.float32
    transpose = np.transpose

    w, h, l = bbox3d[..., 3:6] / 2.  # half of witdh, height and length

    local_points = asarray(  # 8 x 4 matrix
        [
            [ l,  w, -h, 1],  # front left bottom
            [ l, -w, -h, 1],  # front right bottom
            [ l, -w,  h, 1],  # front right top
            [ l,  w,  h, 1],  # front left top
            [-l,  w, -h, 1],  # rear left bottom
            [-l, -w, -h, 1],  # rear right bottom
            [-l, -w,  h, 1],  # rear right top
            [-l,  w,  h, 1],  # rear left top
        ],
        dtype=dtype
    )

    world_points = (trans_matrix @ transpose(local_points)).T[..., :3]  # 8 x 3 matrix
    return world_points


def get_line_func(x1, y1, x2, y2, func_type='xy'):
    """ Get the function of a line.
    Args:
        x1, y1, x2, y2: Two points on the line.
        type: 'xy' or 'yx'. Default is 'xy'.
    Return:
        Functions that can calculate the y or x value by given x or y value.

    """
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    if func_type == 'xy':
        return lambda x: k * x + b
    elif func_type == 'yx':
        return lambda y: (y - b) / k
    else:
        raise ValueError('func_type must be "xy" or "yx".')


def vcs_bbox3d_to_ccs_corners(bbox3d:np.ndarray, calib:Calibration, order='xyz'):
    """
    Args:
        coordinate: [nx7] array [x, y, z, w, h, l, roll, pitch, yaw] in VCS.
        calib: Calibration object
        order: Rotation order. Default is 'xyz'.
        with_depth: If True, return ics corners with depth. Default is False.
    """
    ccs_bbox3d_corners = vcs_to_ccs_points(vcs_bbox3d_to_vcs_corners(bbox3d, order=order), calib)
    return ccs_bbox3d_corners


def vcs_bbox3d_to_ics_corners(bbox3d:np.ndarray, calib:Calibration, order='xyz', with_depth:bool=False, new_intrinsic_calib:Calibration=None):
    """
    Args:
        coordinate: [nx7] array [x, y, z, w, h, l, roll, pitch, yaw] in VCS.
        calib: Calibration object
        order: Rotation order. Default is 'xyz'.
        with_depth: If True, return ics corners with depth. Default is False.
    """
    ccs_bbox3d_corners = vcs_bbox3d_to_ccs_corners(bbox3d, calib, order=order)
    ics_bbox3d_corners = ccs_to_ics_points(ccs_bbox3d_corners, calib, new_intrinsic_calib=new_intrinsic_calib)

    if with_depth:
        return ccs_bbox3d_corners, ics_bbox3d_corners
    else:
        return ccs_bbox3d_corners, ics_bbox3d_corners[..., :2]


def euler_angle_to_rotation_matrix(angle:np.ndarray, translation:np.ndarray=None, order='xyz'):
    """
    Args:
        angle: [..., [roll, pitch, yaw]]
        translation: [..., [tx, ty, tz]]
        order: Rotation order as string. (x(roll), y(pitch), z(yaw))
    Return:
        rotation_translation_matrix: [3, 4] rotation and translation matrix. [R|t]
    """

    r = angle[..., 0]  # roll
    p = angle[..., 1]  # pitch
    y = angle[..., 2]  # yaw

    cos = np.cos
    sin = np.sin
    zeros = np.zeros
    dtype = angle.dtype
    cr = cos(r)
    sr = sin(r)
    cp = cos(p)
    sp = sin(p)
    cy = cos(y)
    sy = sin(y)

    a_shape = angle.shape

    rot = dict()
    rot['x'] = zeros(a_shape + (3,), dtype=dtype)
    rot['x'][..., 0, 0] = 1
    rot['x'][..., 1, 1] = cr
    rot['x'][..., 1, 2] = -sr
    rot['x'][..., 2, 1] = sr
    rot['x'][..., 2, 2] = cr

    rot['y'] = zeros(a_shape + (3,), dtype=dtype)
    rot['y'][..., 0, 0] = cp
    rot['y'][..., 0, 2] = sp
    rot['y'][..., 1, 1] = 1
    rot['y'][..., 2, 0] = -sp
    rot['y'][..., 2, 2] = cp

    rot['z'] = zeros(a_shape + (3,), dtype=dtype)
    rot['z'][..., 0, 0] = cy
    rot['z'][..., 0, 1] = -sy
    rot['z'][..., 1, 0] = sy
    rot['z'][..., 1, 1] = cy
    rot['z'][..., 2, 2] = 1

    rotation_matrix = rot[order[0]] @ rot[order[1]] @ rot[order[2]]
    rotation_translation_matrix = zeros(a_shape[:len(a_shape) - 1] + (4, 4), dtype=dtype)
    rotation_translation_matrix[..., -1, -1] = 1.
    rotation_translation_matrix[..., :3, :3] = rotation_matrix
    if translation is not None:
        rotation_translation_matrix[..., :3, 3] = translation

    return rotation_translation_matrix


def quaternion_to_rotation_matrix(quaternion:np.ndarray, translation:np.ndarray=None):
    """ Covert a quaternion into a full three-dimensional rotation matrix.

    Args:
        Q: A 4 element ar   ray representing the quaternion (w, x, y, z)

    Return:
        rotation_translation_matrix: [3, 4] rotation and translation matrix. [R|t]
    """
    norm = np.linalg.norm(quaternion)
    if not np.isclose(norm, 1.0, atol=1e-12):
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError("Normalize quaternioning with norm=0 would lead to division by zero.")
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

    rotation_translation_matrix = np.zeros([4, 4], dtype=np.float32)
    rotation_translation_matrix[-1, -1] = 1.
    rotation_translation_matrix[:3, :3] = rotation_matrix
    if translation is not None:
        rotation_translation_matrix[:3, 3] = translation

    return rotation_translation_matrix


def rotation_matrix_to_euler_angle(R, order='xyz') :
    """ Covert a rotation matrix into euler angle.
    Args:
        R: [3, 3] rotation matrix
        order: Rotation order as string. (x(roll), y(pitch), z(yaw))
    """
    if order == 'zyx':
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])
    elif order == 'xyz':
        odd = False
        res = [0, 0, 0]
        res[0] = np.arctan2(R[1, 2], R[2, 2])
        c2 = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)
        if (odd and res[0] < 0) or (not odd and res[0] > 0):
            if res[0] > 0:
                res[0] = res[0] - np.pi
            else:
                res[0] = res[0] + np.pi
            res[1] = np.arctan2(-R[0, 2], -c2)
        else:
            res[1] = np.arctan2(-R[0, 2], c2)
        s1 = np.sin(res[0])
        c1 = np.cos(res[0])
        res[2] = np.arctan2(s1 * R[2, 0] - c1 * R[1, 0], c1 * R[1, 1] - s1 * R[2, 1])
        return np.array(res)
    else:
        raise NotImplementedError


def quaternion_to_euler_angle(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw # in radians


def calculate_fov_from_calibration(calib:Calibration, image_width:int, image_height:int):
    """Calculate the FOV from the calibration file.
    Args:
        calib: Calibration object
    Returns:
        fov_x: FOV in x direction
        fov_y: FOV in y direction
    """
    # Get the camera intrinsic matrix
    intrinsic_matrix = calib.camera_intrinsic_matrix

    # Calculate the FOV
    fov_x = 2 * np.arctan(image_width / (2 * intrinsic_matrix[0, 0])) * 180 / np.pi
    fov_y = 2 * np.arctan(image_height / (2 * intrinsic_matrix[1, 1])) * 180 / np.pi

    return fov_x, fov_y

def rotation_matrix_to_quaternion(R: np.ndarray):
    # Ensure the matrix is a valid rotation matrix
    assert R.shape == (3, 3)
    
    # Calculate the trace of the matrix
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])