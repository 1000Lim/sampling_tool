"""
Progress tracker for sampling pipelines.
Provides real-time progress updates to the database.
"""
import json
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ProgressInfo:
    """Progress information for a task stage"""
    stage: str  # e.g., "lidar_extract", "image_sample", "raw_convert", "overlay"
    current: int
    total: int
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProgressTracker:
    """
    Tracks progress for sampling tasks and reports to database.
    """

    def __init__(self, update_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize progress tracker.

        Args:
            update_callback: Callback function to report progress updates.
                            Should accept a dict with progress info.
        """
        self.update_callback = update_callback
        self._current_stage: Optional[ProgressInfo] = None
        self._cancelled = False

    def set_stage(self, stage: str, total: int, message: str = ""):
        """
        Set current processing stage.

        Args:
            stage: Stage name (e.g., "lidar_extract", "image_sample", "raw_convert", "overlay")
            total: Total items to process in this stage
            message: Optional message describing the stage
        """
        self._current_stage = ProgressInfo(stage=stage, current=0, total=total, message=message)
        self._report()

    def update(self, current: int, message: str = ""):
        """
        Update progress for current stage.

        Args:
            current: Current item count
            message: Optional progress message
        """
        if self._current_stage:
            self._current_stage.current = current
            if message:
                self._current_stage.message = message
            self._report()

    def increment(self, message: str = ""):
        """
        Increment progress by 1 for current stage.

        Args:
            message: Optional progress message
        """
        if self._current_stage:
            self._current_stage.current += 1
            if message:
                self._current_stage.message = message
            self._report()

    def cancel(self):
        """Mark task as cancelled"""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if task is cancelled"""
        return self._cancelled

    def _report(self):
        """Report current progress via callback"""
        if self.update_callback and self._current_stage:
            self.update_callback(self._current_stage.to_dict())

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress as dict"""
        return self._current_stage.to_dict() if self._current_stage else None
