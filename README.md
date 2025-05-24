# Vids4VLMs

A Python library for processing videos and analyzing them using Vision Language Models (VLMs). This library provides a flexible and efficient way to extract frames from videos and analyze them using various VLM models through the DSPy framework.

## Features

- Efficient video frame extraction using FFmpeg
- Configurable frame extraction parameters (FPS, resolution, time ranges)
- Integration with DSPy for video analysis
- Support for multiple VLM models
- Simple and complex video analysis modules
- Chat-based video analysis capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vids4vlms.git
cd vids4vlms
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have FFmpeg installed on your system.

## Usage

### Basic Frame Extraction

```python
from vlms4vids.extractors.extractor import VideoExtractor4Dspy, VideoExtractorConfig

# Configure frame extraction
config = VideoExtractorConfig(
    fps=1.0,                    # Extract 1 frame per second
    resize_scale=0.5,           # Scale frames to 50% of original size
    start_time='00:00:10.500',  # Start at 10.5 seconds
    end_time='00:00:20.000',    # End at 20 seconds
    max_frames=100              # Maximum frames to extract
)

# Extract frames
video_path = "path/to/your/video.mp4"
frames = VideoExtractor4Dspy()(video_path, config)
```

### Simple Video Analysis

```python
from vlms4vids.analyzers.analyzer import SimpleVideoAnalyzer, AnalyzerConfig

# Configure analyzer
analyzer_config = AnalyzerConfig(
    model_name="openai/o4-mini",
    temperature=1.0,
    max_tokens=20_000
)

# Create analyzer and analyze video
analyzer = SimpleVideoAnalyzer(analyzer_config)
response = analyzer.ask(frames, prompt="What is the main subject of the video?")
print(response)
```

### Custom Analysis Module

```python
from vlms4vids.analyzers.analyzer_complex import VideoModule
from vlms4vids.analyzers.analyzer import VideoSignature

# Use custom module with specific signature
analyzer_with_custom_module = SimpleVideoAnalyzer(
    analyzer_config,
    dspy_module=dspy.ChainOfThought,
    signature=VideoSignature
)
response = analyzer_with_custom_module.ask(frames, prompt="What is happening in the video?")
```

## Configuration

### VideoExtractorConfig

- `resize_dims`: Tuple[int, int] - Target dimensions (width, height)
- `resize_scale`: float - Scale factor between 0-1
- `fps`: float - Frames per second to extract
- `start_time`: str - Start time in HH:MM:SS.xxx format
- `end_time`: str - End time in HH:MM:SS.xxx format
- `max_frames`: int - Maximum number of frames to extract
- `output_path`: str - Path to save extracted frames

### AnalyzerConfig

- `model_name`: str - Name of the VLM model to use
- `temperature`: float - Model temperature for generation
- `max_tokens`: int - Maximum tokens for model output
- `api_key`: str - API key for the model service

## Project Structure

```
vids4vlms/
├── src/
│   └── vlms4vids/
│       ├── analyzers/         # Video analysis modules
│       ├── extractors/        # Frame extraction utilities
│       └── prompts.py         # Default system prompts
├── tests/                     # Test files
├── requirements.txt           # Project dependencies
└── .env                      # Environment variables
```

## Future Improvements

### Live Streaming Support
- Real-time video stream processing capabilities
- Support for various streaming protocols (RTMP, HLS, DASH)
- Low-latency frame extraction and analysis (might be tough)
- Configurable buffer sizes for stream processing
- Event-based analysis triggers (probably just tool calling or MCP support)

### Advanced Model Integration
- Support for non-VLM vision models
  - Segment Anything Model (SAM) for instance segmentation
- Custom model pipeline creation
- Model chaining and ensemble capabilities (ex. highlight all people who come in contact with a football)

### Tool Calling and Automation
- Integration with task automation platforms
- Batch processing capabilities
- Custom action triggers based on analysis results

## License

This project is licensed under the terms included in the LICENSE file (MIT License).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## Acknowledgements

- [DSPy](https://github.com/stanfordnlp/dspy)
- [FFmpeg](https://ffmpeg.org/)
