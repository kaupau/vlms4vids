from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Iterable
from extractors.utils import VideoProcessingConfig, extract_frames_from_video
import numpy as np
import dspy
from PIL import Image

@dataclass
class VideoExtractorConfig:
    """Configuration for frame extraction"""
    resize_dims: Optional[Tuple[int, int]] = None  # (width, height)
    resize_scale: Optional[float] = None  # Scale factor between 0-1
    fps: Optional[float] = 1.0  # Frames per second to extract
    start_time: Optional[str] = None  # Start time in HH:MM:SS.xxx format
    end_time: Optional[str] = None  # End time in HH:MM:SS.xxx format
    max_frames: Optional[int] = None  # Maximum number of frames
    output_path: Optional[str] = None  # Path to save frames if needed

class VideoExtractor(Protocol):
    def __call__(self, video_path: str, cfg: VideoExtractorConfig) -> Iterable[np.ndarray]:
        ...

class DefaultVideoExtractor:
    """Default implementation of frame extraction using utils.py"""
    def __call__(self, video_path: str, cfg: VideoExtractorConfig) -> Iterable[np.ndarray]:
        # Convert ExtractorConfig to VideoProcessingConfig
        video_cfg = VideoProcessingConfig(
            resize_dims=cfg.resize_dims,
            resize_scale=cfg.resize_scale,
            fps=cfg.fps,
            start_time=cfg.start_time,
            end_time=cfg.end_time,
            max_frames=cfg.max_frames
        )
        
        # Extract frames using the utility function
        return extract_frames_from_video(
            input_path=video_path,
            output_path=cfg.output_path,
            config=video_cfg
        )
    
class VideoExtractor4Dspy(DefaultVideoExtractor):
    """DSPy implementation of frame extraction"""
    def __call__(self, video_path: str, cfg: VideoExtractorConfig) -> Iterable[dspy.Image]:
        frames = super().__call__(video_path, cfg)
        video_frames = [Image.fromarray(np.uint8(frame)) for frame in frames]
        return [dspy.Image.from_PIL(frame) for frame in video_frames]