import dspy
from typing import Any, Iterable, Type, Optional, Union
import pydantic
from src.analyzers.analyzer import DSPyAnalyzer, VideoSignature, AnalyzerConfig
from vids4vlms.prompts import DEFAULT_SYSTEM_PROMPT

class VideoModule(dspy.Module):
    def __init__(
        self,
        signature: Type[dspy.Signature],
        **config,
    ):
        signature = signature.prepend(
            name="frames",
            field=dspy.InputField(desc="A list of sequential frames from a video"),
            type_=list[dspy.Image]
        )
        super().__init__()
        self.predict = dspy.ChainOfThought(signature, **config)

    def __call__(self, frames: Iterable[dspy.Image], **kwargs) -> Any:
        return self.predict(frames, **kwargs)
