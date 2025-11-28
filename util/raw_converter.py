"""
Raw image converter utility for ACLM data.
Extracted from surf_raw_image_converter_ui.py v1.10 for ACLM RAW14 conversion.

Version: v1.10
Changes from v1.8:
  - Updated ACLM crop region: CROP_TOP changed from 6 to 4
"""
import io
import numpy as np
from PIL import Image
from enum import Enum
from typing import Optional

try:
    import scipy.interpolate as spi
    import cv2
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class RawOutputFormat(Enum):
    GRAY = "gray"
    RGB = "rgb"


class AclmIspProcessor:
    """ACLM-specific ISP pipeline for RGB conversion"""

    DEPWL_X = np.array([0, 256, 2304, 4352, 6400, 8448, 10496, 12544, 14592, 16896,
                        18944, 20992, 23040, 25088, 27136, 29184, 31232, 33792, 35840,
                        37888, 39936, 41984, 44032, 46080, 48128, 50176, 52224, 54272,
                        56320, 58368, 60416, 62464, 65535], dtype=np.float32)
    DEPWL_Y = np.array([0, 64, 576, 1088, 2112, 4160, 6208, 10304, 14400, 22592,
                        30784, 38976, 47168, 63552, 79936, 96320, 112704, 129088, 161856,
                        194624, 260160, 325696, 391232, 522304, 653376, 784448, 1046592,
                        1308736, 1570880, 2095168, 2619456, 3143744, 4194303], dtype=np.float32)

    def __init__(self):
        if not DEPS_AVAILABLE:
            raise ImportError("ACLM RGB conversion requires scipy and opencv-python: pip install scipy opencv-python")
        self._depwl_func = spi.interp1d(self.DEPWL_X, self.DEPWL_Y, kind='linear')

    def unpack_raw14(self, data: bytes, height: int, width: int) -> np.ndarray:
        """Unpack MIPI RAW14 format"""
        packed = np.frombuffer(data, dtype=np.uint8)
        unpacked = np.zeros(height * width, dtype=np.uint16)

        packed = packed[:height * width * 7 // 4].reshape(-1, 7).astype(np.uint16)
        unpacked[0::4] = packed[:, 0] + ((packed[:, 1] & 0x3F) << 8)
        unpacked[1::4] = ((packed[:, 1] & 0xC0) >> 6) + (packed[:, 2] << 2) + ((packed[:, 3] & 0x0F) << 10)
        unpacked[2::4] = ((packed[:, 3] & 0xF0) >> 4) + (packed[:, 4] << 4) + ((packed[:, 5] & 0x03) << 12)
        unpacked[3::4] = ((packed[:, 5] & 0xFC) >> 2) + (packed[:, 6] << 6)
        unpacked = unpacked << 2  # align to 16-bit MSB

        return unpacked.reshape(height, width)

    def _simple_demosaic(self, raw: np.ndarray) -> np.ndarray:
        """Simple demosaic for GRBG bayer pattern"""
        h, w = raw.shape
        # GRBG pattern: G at (0,0), R at (0,1), B at (1,0), G at (1,1)
        g = (raw[0::2, 0::2] + raw[1::2, 1::2]) / 2
        r = raw[0::2, 1::2]
        b = raw[1::2, 0::2]

        r_up = cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)
        g_up = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        b_up = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.stack([r_up, g_up, b_up], axis=0)

    def process(self, raw: np.ndarray, black_level: int, dgain: float = 1.5) -> np.ndarray:
        """
        Process raw image to RGB with ISP pipeline.

        Args:
            raw: Raw image array (uint16)
            black_level: Black level value
            dgain: Digital gain (default: 1.5)

        Returns:
            RGB image (uint8, HWC format)
        """
        raw = raw.astype(np.float32)

        # De-piecewise linear
        raw = self._depwl_func(raw)

        # Black level correction
        bl_depwl = float(self._depwl_func(black_level))
        wl_depwl = float(self._depwl_func(65535))
        raw = raw - bl_depwl
        raw = raw * (wl_depwl / (wl_depwl - bl_depwl))
        raw = np.clip(raw, 0.0, wl_depwl)

        # Auto white balance
        r_mean = raw[::2, ::2].mean()
        g_mean = (raw[::2, 1::2].mean() + raw[1::2, ::2].mean()) / 2
        b_mean = raw[1::2, 1::2].mean()
        wb_r, wb_b = g_mean / r_mean, g_mean / b_mean
        raw[::2, ::2] *= wb_r
        raw[1::2, 1::2] *= wb_b

        # Demosaic
        rgb = self._simple_demosaic(raw)

        # Normalize and apply auto-brightness
        rgb = rgb / wl_depwl
        lo = float(np.percentile(rgb, 0.5))
        hi = float(np.percentile(rgb, 99.5))
        if hi > lo + 1e-6:
            rgb = (rgb - lo) / (hi - lo)
        rgb = np.clip(rgb, 0.0, 1.0)

        # Digital gain for low-light boost
        rgb = np.clip(rgb * dgain, 0.0, 1.0)

        # Gamma correction
        rgb = np.power(rgb, 0.4)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)

        # CHW -> HWC, RGB order
        return rgb.transpose(1, 2, 0)


class RawImageProcessor:
    """Gray image processing utilities"""

    @staticmethod
    def read_mipi_raw14(data: bytes, width: int, height: int, stride: int) -> np.ndarray:
        """Vectorized RAW14 unpacking: 7 bytes -> 4 pixels"""
        raw = np.frombuffer(data, dtype=np.uint8)[:height * stride].reshape(height, stride)

        # Number of pixel quads (4 pixels per 7 bytes)
        n_quads = width // 4
        n_bytes = n_quads * 7

        # Extract byte groups (7 bytes each)
        groups = raw[:, :n_bytes].reshape(height, n_quads, 7)
        b = groups.astype(np.uint16)

        # Decode 4 pixels from 7 bytes
        p0 = b[:, :, 0] | ((b[:, :, 1] & 0x3F) << 8)
        p1 = (b[:, :, 1] >> 6) | (b[:, :, 2] << 2) | ((b[:, :, 3] & 0x0F) << 10)
        p2 = (b[:, :, 3] >> 4) | (b[:, :, 4] << 4) | ((b[:, :, 5] & 0x03) << 12)
        p3 = (b[:, :, 5] >> 2) | (b[:, :, 6] << 6)

        # Interleave pixels
        img = np.empty((height, n_quads * 4), dtype=np.uint16)
        img[:, 0::4] = p0
        img[:, 1::4] = p1
        img[:, 2::4] = p2
        img[:, 3::4] = p3

        # Handle remaining pixels
        remainder = width % 4
        if remainder > 0:
            extra_start = n_bytes
            extra = np.zeros((height, remainder), dtype=np.uint16)
            for h in range(height):
                bitbuf, bitcount, idx = 0, 0, 0
                for i in range(extra_start, min(extra_start + 7, stride)):
                    bitbuf |= int(raw[h, i]) << bitcount
                    bitcount += 8
                    while bitcount >= 14 and idx < remainder:
                        extra[h, idx] = bitbuf & 0x3FFF
                        bitbuf >>= 14
                        bitcount -= 14
                        idx += 1
            img = np.concatenate([img, extra], axis=1)

        return img[:, :width]

    @staticmethod
    def apply_black_white_level(img: np.ndarray, black: int, white: int) -> np.ndarray:
        """Apply black and white level correction"""
        corrected = np.clip(img.astype(np.int32) - black, 0, white - black)
        return corrected.astype(np.float32) / max(1, white - black)

    @staticmethod
    def auto_brightness(img01: np.ndarray, low_perc: float = 0.5, high_perc: float = 99.5) -> np.ndarray:
        """Auto brightness adjustment"""
        lo, hi = float(np.percentile(img01, low_perc)), float(np.percentile(img01, high_perc))
        if hi <= lo + 1e-6:
            return img01
        return np.clip((img01 - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def crop_rect(img: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
        """Crop rectangle from image"""
        return img[top:top + height, left:left + width]


class AclmRawConverter:
    """ACLM RAW14 to JPG converter"""

    # ACLM RAW14 specs (v1.10)
    WIDTH = 3848
    HEIGHT = 2172
    STRIDE = 6734
    TARGET_WIDTH = 3840
    TARGET_HEIGHT = 2160
    CROP_TOP = 4  # v1.10: updated from 6 to 4
    CROP_LEFT = 4
    BLACK_LEVEL = 256
    WHITE_LEVEL = 16383

    def __init__(self, output_format: RawOutputFormat = RawOutputFormat.GRAY, dgain: float = 1.5):
        """
        Initialize converter.

        Args:
            output_format: Output format (GRAY or RGB)
            dgain: Digital gain for RGB conversion (default: 1.5)
        """
        self.output_format = output_format
        self.dgain = dgain
        self._isp: Optional[AclmIspProcessor] = None

        if output_format == RawOutputFormat.RGB:
            self._isp = AclmIspProcessor()

    def convert_raw_to_jpg_bytes(self, raw_data: bytes) -> bytes:
        """
        Convert raw data to JPG bytes.

        Args:
            raw_data: Raw image bytes

        Returns:
            JPG image bytes
        """
        if self.output_format == RawOutputFormat.RGB:
            return self._convert_rgb(raw_data)
        else:
            return self._convert_gray(raw_data)

    def _convert_gray(self, raw_data: bytes) -> bytes:
        """Convert to grayscale JPG"""
        # Unpack RAW14
        img = RawImageProcessor.read_mipi_raw14(
            raw_data, self.WIDTH, self.HEIGHT, self.STRIDE
        )

        # Crop to target size
        img = RawImageProcessor.crop_rect(
            img, self.CROP_TOP, self.CROP_LEFT,
            self.TARGET_HEIGHT, self.TARGET_WIDTH
        )

        # Apply black/white level and normalize
        norm01 = RawImageProcessor.apply_black_white_level(
            img, self.BLACK_LEVEL, self.WHITE_LEVEL
        )

        # Auto brightness
        vis01 = RawImageProcessor.auto_brightness(norm01, 0.5, 99.5)

        # Convert to 8-bit
        gray8 = (vis01 * 255.0 + 0.5).astype(np.uint8)

        # Save as JPG
        pil_img = Image.fromarray(gray8, mode='L')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()

    def _convert_rgb(self, raw_data: bytes) -> bytes:
        """Convert to RGB JPG using ISP pipeline"""
        # Unpack RAW14
        raw = self._isp.unpack_raw14(raw_data, self.HEIGHT, self.WIDTH)

        # Crop to target size
        raw = RawImageProcessor.crop_rect(
            raw, self.CROP_TOP, self.CROP_LEFT,
            self.TARGET_HEIGHT, self.TARGET_WIDTH
        )

        # Process with ISP
        rgb8 = self._isp.process(raw, self.BLACK_LEVEL, self.dgain)

        # Save as JPG
        pil_img = Image.fromarray(rgb8, mode='RGB')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
