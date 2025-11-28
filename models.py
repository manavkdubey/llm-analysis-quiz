"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any


class QuizRequest(BaseModel):
    """Request model for quiz endpoint."""
    email: str = Field(..., description="Student email ID")
    secret: str = Field(..., description="Student-provided secret")
    url: str = Field(..., description="Quiz URL to solve")
    # Allow other fields
    class Config:
        extra = "allow"


class QuizResponse(BaseModel):
    """Response model from quiz submission endpoint."""
    correct: bool = Field(..., description="Whether the answer is correct")
    url: Optional[str] = Field(None, description="Next quiz URL if available")
    reason: Optional[str] = Field(None, description="Reason for incorrect answer")
    class Config:
        extra = "allow"


class AnswerPayload(BaseModel):
    """Payload for submitting answers."""
    email: str
    secret: str
    url: str
    answer: Union[bool, int, float, str, Dict[str, Any]] = Field(
        ..., description="The answer (can be various types)"
    )

