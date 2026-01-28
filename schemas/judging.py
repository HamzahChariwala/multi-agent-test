"""Judging phase schemas for council members to evaluate candidate answers."""

from typing import Dict, List
from pydantic import BaseModel, Field


class JudgingRequest(BaseModel):
    """Request schema for judging phase."""
    
    task_prompt: str = Field(..., description="The original task")
    candidates: Dict[str, str] = Field(
        ...,
        description="Map of candidate_id to candidate answer"
    )
    rubric: str = Field(
        default="Evaluate based on correctness, completeness, clarity, and feasibility.",
        description="Evaluation rubric for judging"
    )
    request_id: str = Field(..., description="Unique request identifier for tracking")


class JudgingOutput(BaseModel):
    """Output schema for judging phase from a council member."""
    
    scores: Dict[str, float] = Field(
        ...,
        description="Map of candidate_id to score (0.0 to 1.0)"
    )
    ranking: List[str] = Field(
        ...,
        description="Ordered list of candidate_ids from best to worst"
    )
    top_reasoning: Dict[str, str] = Field(
        ...,
        description="Reasoning for top 3 candidates (candidate_id -> rationale)"
    )
    judge_id: str = Field(..., description="ID of the member who judged")
    
    class Config:
        json_schema_extra = {
            "example": {
                "scores": {
                    "member_1": 0.85,
                    "member_2": 0.72,
                    "member_3": 0.91
                },
                "ranking": ["member_3", "member_1", "member_2"],
                "top_reasoning": {
                    "member_3": "Most comprehensive and well-structured solution",
                    "member_1": "Good approach but missing edge cases",
                    "member_2": "Correct but less detailed"
                },
                "judge_id": "member_1"
            }
        }

