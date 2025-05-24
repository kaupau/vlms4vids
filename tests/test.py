from src.analyzers.analyzer import SimpleVideoAnalyzer, VideoSignature
from src.analyzers.analyzer_complex import VideoModule
# how do i unify the imports?
from src.analyzers.analyzer import AnalyzerConfig
from src.extractors.extractor import VideoExtractor4Dspy, VideoExtractorConfig
import dspy

video_path = "test_videos/0_tr76.mp4"
frames = VideoExtractor4Dspy()(video_path, VideoExtractorConfig(fps=1, resize_scale=0.1))
analyzer_config = AnalyzerConfig(model_name="openai/o4-mini", temperature=1.0, max_tokens=20_000)

# Simple Video Analyzer
analyzer = SimpleVideoAnalyzer(analyzer_config)
response = analyzer.ask(frames, prompt="What is the main subject of the video?")
print(response)

# Simple Video Analyzer with custom Module
analyzer_with_custom_module = SimpleVideoAnalyzer(analyzer_config, dspy_module=dspy.ChainOfThought, signature=VideoSignature)
response = analyzer_with_custom_module.ask(frames, prompt="What is the main subject of the video?")
print(response)

# Complex Video Analyzer
# another call - so that it can be used as a function and we provide the frames as an argument + other parameters

# TODO: add support for multiple videos with different prompts


