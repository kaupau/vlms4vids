from dataclasses import dataclass
from typing import Any, Iterable, Protocol
import dspy
from vlms4vids.prompts import DEFAULT_SYSTEM_PROMPT

@dataclass
class AnalyzerConfig:
    """Configuration for DSPy analysis"""
    model_name: str = "openai/o4-mini"
    temperature: float = 1.0
    max_tokens: int = 20_000
    api_key: str = None

class VideoSignature(dspy.Signature):
    """You are a helpful assistant that can answer questions about a video"""
    frames: list[dspy.Image] = dspy.InputField(desc="A list of sequential frames from a video")
    system_prompt: str = dspy.InputField(desc="Analyze the video based on the prompt", default=DEFAULT_SYSTEM_PROMPT)
    answer: str = dspy.OutputField(desc="An answer to the prompt")

class VideoChatSignature(dspy.Signature):
    """You are a helpful assistant that can answer questions about a video"""
    frames: list[dspy.Image] = dspy.InputField(desc="A list of sequential frames from a video")
    chat_history: list[dict[str, str]] = dspy.InputField(desc="A list of previous messages between the user and the assistant")
    answer: str = dspy.OutputField(desc="An answer to the question")

    

class Analyzer(Protocol):
    def __call__(self, frames: Iterable[dspy.Image], cfg: AnalyzerConfig) -> Any:
        ...

class DSPyAnalyzer(Analyzer):
    def __init__(self, cfg: AnalyzerConfig):
        lm = dspy.LM(cfg.model_name, api_key=cfg.api_key, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        dspy.configure(lm=lm)

    def change_config(self, cfg: AnalyzerConfig):
        lm = dspy.LM(cfg.model_name, api_key=cfg.api_key, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        dspy.configure(lm=lm)
        # https://github.com/stanfordnlp/dspy/issues/1589
        # above link is a bug in dspy - check if it's fixed
    
class SimpleVideoAnalyzer(DSPyAnalyzer):
    def __init__(self, cfg: AnalyzerConfig, dspy_module: dspy.Module = dspy.ChainOfThought, signature: dspy.Signature = VideoSignature):
        super().__init__(cfg)
        self.dspy_module = dspy_module

        # add frames to signature if not present
        if "frames" not in signature.input_fields:
            signature = signature.prepend(
                name="frames",
                field=dspy.InputField(desc="A list of sequential frames from a video"),
                type_=list[dspy.Image]
            )
        self.signature = signature

        # add chat_history to signature if not present
        if "chat_history" or "messages" not in signature.input_fields:
            signature = signature.prepend(
                name="chat_history",
                field=dspy.InputField(desc="A list of previous messages between the user and the assistant"),
                type_=list[dict[str, str]]
            )
        self.chat_signature = signature

    def ask(self, frames: Iterable[dspy.Image], prompt: str) -> Any:
        return self.dspy_module(self.signature)(frames=frames, system_prompt=prompt)
    
    def chat(self, frames: Iterable[dspy.Image], messages: list[dict[str, str]]) -> Any:
        return self.dspy_module(self.chat_signature)(frames=frames, chat_history=messages)
    



# Cases:
# - Analyze a video
# - Analyze a video with custom prompt + input output fields / signature
# - Chat with a video

# support:
# call(frames, input pydantic, output pydantic)
# call(frames, input str, output str)
# call(frames, dspy signature)
# call(frames, dspy signature, dspy module)

