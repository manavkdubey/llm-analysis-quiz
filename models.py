from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any


class QuizRequest(BaseModel):
    email: str = Field(..., description="Student email ID")
    secret: str = Field(..., description="Student-provided secret")
    url: str = Field(..., description="Quiz URL to solve")
    class Config:
        extra = "allow"


class QuizResponse(BaseModel):
    correct: bool = Field(..., description="Whether the answer is correct")
    url: Optional[str] = Field(None, description="Next quiz URL if available")
    reason: Optional[str] = Field(None, description="Reason for incorrect answer")
    class Config:
        extra = "allow"


class AnswerPayload(BaseModel):
    email: str
    secret: str
    url: str
    answer: Union[bool, int, float, str, Dict[str, Any]] = Field(
        ..., description="The answer (can be various types)"
    )

