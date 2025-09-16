"""
This module stores the constants used in the project.
"""
import os
import enum

class DataType(enum.Enum):
    SURF='surf'
    VALEO='valeo'

OVERLAY_DIR = os.path.abspath('overlay_images')
SAMPLING_DIR = os.path.abspath('sampling')

