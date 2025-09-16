from dataclasses import dataclass
from typing import List
import csv
from loguru import logger

SAMPLE_FRAME_HEADER = ['name', 'frame_id', 'frame_timestamp', 'lidar_timestamp', 'wheel_speed']
SAMPLE_CSV_NAME = 'sample.csv'

@dataclass
class SampleFrame:
    frame_id: int
    frame_timestamp: int    
    lidar_timestamp: int
    wheel_speed: int = 0
    name: str = ''
    
    def __str__(self):
        """
        If the name is not empty, return the name else return the frame id, frame timestamp, and matched lidar timestamp.
        """
        if self.name:
            return self.name
        else:
            return f"Frame ID: {self.frame_id}, Frame Timestamp: {self.frame_timestamp}, Matched Lidar Timestamp: {self.matched_lidar_timestamp}"
        

def write_sample_frames_to_csv(sample_frames: List[SampleFrame], csv_file_path: str = SAMPLE_CSV_NAME) -> None:
    """
    Write the sample frames to the csv file.
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SAMPLE_FRAME_HEADER)
        writer.writeheader()
        for sample_frame in sample_frames:
            writer.writerow({'name': sample_frame.name, 'frame_id': sample_frame.frame_id, 'frame_timestamp': sample_frame.frame_timestamp, 'lidar_timestamp': sample_frame.lidar_timestamp, 'wheel_speed': sample_frame.wheel_speed})
            
def read_sample_frames_from_csv(csv_file_path: str = SAMPLE_CSV_NAME):
    """
    Read the sample frames from the csv file.
    """
    sample_frames: List[SampleFrame] = []

    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sample_frames.append(SampleFrame(frame_id=int(row['frame_id']), frame_timestamp=int(row['frame_timestamp']), lidar_timestamp=int(row['lidar_timestamp']), wheel_speed=int(row['wheel_speed']), name=row['name']))
        
        return sample_frames
    except FileNotFoundError:
        logger.error(f"Sample Frame File not found: {csv_file_path}")
        return sample_frames
    except Exception as e:
        logger.error(f"Failed to read the sample frame file: {csv_file_path}, error: {e}")
        return sample_frames
    