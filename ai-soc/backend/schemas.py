from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any

class AlertBase(BaseModel):
    attack: str
    confidence: float
    severity: str

class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    time: datetime

    class Config:
        from_attributes = True

class PredictRequest(BaseModel):
    # Expecting exactly 80 numeric features based on CICIDS2017 training
    features: List[float] = Field(..., min_items=80, max_items=80)

class PredictResponse(BaseModel):
    attack_name: str
    confidence_score: float
    shap_values: Optional[Dict[str, float]] = None
    alert_id: Optional[int] = None
