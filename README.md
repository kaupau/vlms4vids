# vids4vlms

A Python library for seamlessly interfacing videos with Vision Language Models (VLMs). This library provides tools for video frame extraction, processing, and analysis using modern VLMs.

## Features

- 🎥 **Efficient Video Processing**: Extract and process video frames with configurable parameters
- 🔄 **Flexible Frame Extraction**: Control FPS, resolution, time ranges, and more
- 🤖 **VLM Integration**: Built-in support for vision language models through DSPy
- 💬 **Interactive Analysis**: Support for both single-query analysis and chat-based interaction
- ⚡ **Optimized Performance**: Efficient frame handling and processing using numpy and PIL

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from vids4vlms.extractors.extractor import DSPyExtractor, ExtractorConfig
from vids4vlms.analyzers.analyzer import SimpleVideoAnalyzer, AnalyzerConfig

# Configure and extract frames from a video
extractor_config = ExtractorConfig(
    fps=1.0,                    # Extract 1 frame per second
    resize_scale=0.5,           # Scale frames to 50% of original size
    max_frames=100              # Maximum number of frames to extract
)

# Extract frames
extractor = DSPyExtractor()
frames = extractor("path/to/video.mp4", extractor_config)

# Configure and create analyzer
analyzer_config = AnalyzerConfig(
    model_name="openai/o4-mini",
    temperature=1.0,
    max_tokens=20_000
)

# Analyze video
analyzer = SimpleVideoAnalyzer(analyzer_config)
response = analyzer.ask(frames, prompt="What is happening in this video?")
print(response)
```

## Core Components

### 1. Extractors

- `ExtractorConfig`: Configure frame extraction parameters
- `DefaultExtractor`: Basic frame extraction functionality
- `DSPyExtractor`: DSPy-compatible frame extraction

### 2. Analyzers

- `SimpleVideoAnalyzer`: Basic video analysis capabilities
- `VideoModule`: Custom DSPy module for video analysis
- Support for both single queries and chat-based interaction

## Configuration Options

### Extractor Configuration

```python
ExtractorConfig(
    resize_dims=(640, 480),     # Exact dimensions (width, height)
    resize_scale=0.5,           # Scale factor (0-1)
    fps=1.0,                    # Frames per second
    start_time='00:00:10.500',  # Start time (HH:MM:SS.xxx)
    end_time='00:00:20.000',    # End time (HH:MM:SS.xxx)
    max_frames=100,             # Maximum frames to extract
    output_path='path/to/save'  # Optional path to save frames
)
```

### Analyzer Configuration

```python
AnalyzerConfig(
    model_name="openai/o4-mini",  # VLM model to use
    temperature=1.0,              # Model temperature
    max_tokens=20_000            # Maximum tokens for response
)
```

## Advanced Usage

### Chat-based Analysis

```python
# Create chat history
messages = [
    {"role": "user", "content": "What objects are visible in the video?"},
    {"role": "assistant", "content": "I can see..."},
    {"role": "user", "content": "What happens next?"}
]

# Get response
response = analyzer.chat(frames, messages)
```

### Custom DSPy Modules

```python
import dspy
response = analyzer.ask(
    frames,
    prompt="Describe the video in detail",
    dspy_module=dspy.ChainOfThought
)
```

## Requirements

- python-dotenv
- dspy
- numpy
- pillow
- pydantic

## License

[License information not found in codebase]

## Contributing

[Contribution guidelines not found in codebase] 