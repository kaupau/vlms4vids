from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import ffmpeg

@dataclass
class VideoProcessingConfig:
    """Configuration for video frame extraction"""
    resize_dims: Optional[Tuple[int, int]] = None  # (width, height)
    resize_scale: Optional[float] = None  # Scale factor between 0-1 to resize frames
    fps: Optional[float] = 1.0  # Extract n frames per second
    start_time: Optional[str] = None  # Start time in HH:MM:SS.xxx format
    end_time: Optional[str] = None  # End time in HH:MM:SS.xxx format
    max_frames: Optional[int] = None  # Maximum number of frames to extract

def transform_video_stream(
    stream: ffmpeg.Stream,
    config: VideoProcessingConfig,
    orig_width: int,
    orig_height: int
) -> Tuple[ffmpeg.Stream, int, int]:
    """
    Apply transformations (resize, fps) to video stream
    
    Args:
        stream: Input ffmpeg stream
        config: Processing configuration
        orig_width: Original video width
        orig_height: Original video height
        
    Returns:
        Tuple of (transformed stream, new width, new height)
    """
    # Calculate dimensions based on resize_scale if provided
    if config.resize_scale is not None:
        if not 0 < config.resize_scale <= 1:
            raise ValueError("resize_scale must be between 0 and 1")
        width = int(orig_width * config.resize_scale)
        height = int(orig_height * config.resize_scale)
    else:
        width = config.resize_dims[0] if config.resize_dims else orig_width
        height = config.resize_dims[1] if config.resize_dims else orig_height
    
    # Apply filters one by one instead of joining them
    if config.fps:
        stream = stream.filter('fps', fps=config.fps)
    
    if config.resize_scale or config.resize_dims:
        stream = stream.filter('scale', width, height)
        
    return stream, width, height

def extract_frames_from_buffer(
    buffer: bytes,
    frame_size: int,
    width: int,
    height: int,
    max_frames: Optional[int] = None,
    output_path: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract frames from raw video buffer
    
    Args:
        buffer: Raw video buffer from ffmpeg
        frame_size: Size of each frame in bytes
        width: Frame width
        height: Frame height
        max_frames: Maximum number of frames to extract
        output_path: Optional path to save frames
        
    Returns:
        List of numpy arrays containing frames
    """
    frames = []
    
    for i in range(0, len(buffer), frame_size):
        if max_frames and len(frames) >= max_frames:
            break
            
        frame_bytes = buffer[i:i + frame_size]
        if len(frame_bytes) != frame_size:
            break
            
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        frames.append(frame)
        
        if output_path:
            img = Image.fromarray(frame)
            img.save(f"{output_path}/frame_{len(frames):06d}.jpg")
    
    return frames

def extract_frames_from_video(
    input_path: str,
    output_path: Optional[str] = None,
    config: Optional[VideoProcessingConfig] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video file using python-ffmpeg.
    This provides a more Pythonic interface to FFmpeg with better error handling.
    
    Args:
        input_path: Path to input video file
        output_path: Optional path to save extracted frames
        config: VideoProcessingConfig object with processing parameters
        
    Returns:
        List of numpy arrays containing the extracted frames
    """
    config = config or VideoProcessingConfig()
    
    # Start building the ffmpeg stream
    stream = ffmpeg.input(input_path)
    
    # Add time range if specified
    if config.start_time:
        stream = stream.filter('setpts', f'PTS-STARTPTS', start_time=config.start_time)
    if config.end_time:
        duration = _time_to_seconds(config.end_time) - _time_to_seconds(config.start_time or '00:00:00')
        stream = stream.filter('setpts', f'PTS-STARTPTS', duration=duration)
    
    # Get video info
    probe = ffmpeg.probe(input_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    orig_width = int(video_info['width'])
    orig_height = int(video_info['height'])
    
    # Apply transformations
    stream, width, height = transform_video_stream(stream, config, orig_width, orig_height)
    
    # Configure output stream
    stream = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24')
    
    # Run the ffmpeg process
    try:
        out, err = stream.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg._run.Error as e:
        print(f"FFmpeg stderr output:\n{e.stderr.decode('utf-8')}")
        raise
    
    # Extract frames from buffer
    frame_size = width * height * 3
    return extract_frames_from_buffer(
        out,
        frame_size,
        width,
        height,
        config.max_frames,
        output_path
    )

def _time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.xxx time format to seconds"""
    if not time_str:
        return 0
        
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

# Example usage:
"""
config = VideoProcessingConfig(
    resize_dims=(640, 480),     # Exact dimensions
    # OR
    resize_scale=0.5,           # Scale to 50% of original size
    fps=1.0,                    # Extract 1 frame per second
    start_time='00:00:10.500', # Start at 10.5 seconds
    end_time='00:00:20.000',   # End at 20 seconds
    max_frames=100             # Extract maximum 100 frames
)

frames = extract_frames_from_video(
    input_path="path/to/video.mp4",
    output_path="path/to/output/frames",  # Optional
    config=config
)
"""
