"""Data contract schemas for the multi-agent council system."""

from .generation import GenerationOutput, GenerationRequest
from .judging import JudgingOutput, JudgingRequest
from .chairman import ChairmanOutput, ChairmanRequest

__all__ = [
    "GenerationOutput",
    "GenerationRequest",
    "JudgingOutput",
    "JudgingRequest",
    "ChairmanOutput",
    "ChairmanRequest",
]

