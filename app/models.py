from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional


class KeyMapping(BaseModel):
    """Maps user phrase to actual store key"""
    user_phrase: str
    mapped_to: str
    similarity: float  # 0 to 1
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_phrase": "bureau score",
                "mapped_to": "bureau.score",
                "similarity": 0.93
            }
        }
    )


class GenerateRuleRequest(BaseModel):
    """Request for generating a rule"""
    prompt: str = Field(min_length=10, max_length=500)
    context_docs: Optional[List[str]] = Field(
        default=None,
        description="Optional additional policy documents to consider"
    )
    
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Approve if bureau score > 700 and business vintage at least 3 years",
                "context_docs": ["Custom policy: Minimum age 25"]
            }
        }
    )


class GenerateRuleResponse(BaseModel):
    """What we send back after generating a rule"""
    json_logic: Dict[str, Any]
    explanation: str
    used_keys: List[str]
    key_mappings: List[KeyMapping]
    confidence_score: float
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "json_logic": {
                    "and": [
                        {">": [{"var": "bureau.score"}, 700]},
                        {">=": [{"var": "business.vintage_in_years"}, 3]}
                    ]
                },
                "explanation": "Approves applications where bureau score exceeds 700 AND business vintage is at least 3 years.",
                "used_keys": ["bureau.score", "business.vintage_in_years"],
                "key_mappings": [
                    {
                        "user_phrase": "bureau score",
                        "mapped_to": "bureau.score",
                        "similarity": 0.93
                    }
                ],
                "confidence_score": 0.89
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error format"""
    error: str
    detail: Optional[str] = None
    suggestions: Optional[List[Dict[str, Any]]] = None  # suggested keys when nothing matches
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "No matching keys found",
                "detail": "Prompt contains fields not in our key list",
                "suggestions": [
                    {"key": "bureau.score", "similarity": 0.45}
                ]
            }
        }
    )
