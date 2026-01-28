"""Generation phase schemas for council members."""

from typing import List, Optional
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request schema for generation phase."""
    
    task_prompt: str = Field(..., description="The task to be solved")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    request_id: str = Field(..., description="Unique request identifier for tracking")


class GenerationOutput(BaseModel):
    """Output schema for generation phase from a council member."""
    
    answer: str = Field(..., description="The proposed answer to the task")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made while generating the answer"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Potential risks or limitations of this answer"
    )
    member_id: Optional[str] = Field(None, description="ID of the member who generated this")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The solution involves implementing a binary search tree...",
                "assumptions": ["Input data is sorted", "Memory is not constrained"],
                "confidence": 0.85,
                "risks": ["May not scale to large datasets"],
                "member_id": "member_1"
            }
        }

