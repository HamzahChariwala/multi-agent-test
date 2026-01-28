"""Chairman synthesis phase schemas."""

from typing import Dict, List
from pydantic import BaseModel, Field


class ChairmanRequest(BaseModel):
    """Request schema for chairman synthesis phase."""
    
    task_prompt: str = Field(..., description="The original task")
    candidates: Dict[str, str] = Field(
        ...,
        description="Map of candidate_id to candidate answer"
    )
    judgments: List[Dict] = Field(
        ...,
        description="List of judging outputs from all members"
    )
    request_id: str = Field(..., description="Unique request identifier for tracking")


class ChairmanOutput(BaseModel):
    """Output schema for chairman's final synthesis."""
    
    final_answer: str = Field(
        ...,
        description="The synthesized final answer from the council"
    )
    decision_trace: str = Field(
        ...,
        description="Explanation of how the decision was reached"
    )
    selected_candidate_ids: List[str] = Field(
        ...,
        description="List of candidate IDs that contributed to the final answer"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the final answer"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "The optimal solution combines elements from multiple approaches...",
                "decision_trace": "Member 3's structure was chosen as the foundation, enhanced with Member 1's edge case handling...",
                "selected_candidate_ids": ["member_3", "member_1"],
                "confidence": 0.92
            }
        }

