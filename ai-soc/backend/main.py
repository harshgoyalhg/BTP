from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import numpy as np

from . import models, schemas, database, shap_explainer
import sys
import os

# Ensure the predict module can be imported from ml folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ml.predict
from ml.predict import predict_attack

# Create tables if they don't exist
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI SOC API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"status": "AI SOC Backend is running"}

@app.post("/predict", response_model=schemas.PredictResponse)
def predict_endpoint(request: schemas.PredictRequest, db: Session = Depends(get_db)):
    """
    Receives network flow features, predicts if it's an attack, 
    generates SHAP explanations, and logs to the DB if malicious.
    """
    try:
        # Run XGBoost prediction
        result = predict_attack(request.features)
        
        attack_name = result["attack_name"]
        confidence = result["confidence_score"]
        
        shap_values = None
        alert_id = None
        
        # If it's an attack, generate SHAP values and save to database
        if attack_name != "BENIGN":
            # Determine severity based on confidence
            severity = "HIGH" if confidence > 0.9 else "MEDIUM" if confidence > 0.7 else "LOW"
            
            # Generate SHAP explainability
            scaled_features = ml.predict._scaler.transform(np.array(request.features).reshape(1, -1))
            shap_values = shap_explainer.get_shap_values(scaled_features)
            
            # Save alert to DB
            db_alert = models.Alert(
                attack=attack_name,
                confidence=confidence,
                severity=severity
            )
            db.add(db_alert)
            db.commit()
            db.refresh(db_alert)
            alert_id = db_alert.id
            
        return {
            "attack_name": attack_name,
            "confidence_score": confidence,
            "shap_values": shap_values,
            "alert_id": alert_id
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
def get_alerts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve recent alerts for the dashboard."""
    alerts = db.query(models.Alert).order_by(models.Alert.time.desc()).offset(skip).limit(limit).all()
    return alerts

@app.get("/incidents")
def get_incidents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve incidents."""
    incidents = db.query(models.Incident).order_by(models.Incident.created_at.desc()).offset(skip).limit(limit).all()
    return incidents

from sqlalchemy import func

@app.get("/statistics")
def get_stats(db: Session = Depends(get_db)):
    """Get high-level stats for the dashboard."""
    total_alerts = db.query(models.Alert).count()
    high_severity = db.query(models.Alert).filter(models.Alert.severity == "HIGH").count()
    
    # Attack distribution
    distribution = db.query(models.Alert.attack, func.count(models.Alert.id)).group_by(models.Alert.attack).all()
    dist_dict = {row[0]: row[1] for row in distribution}
    
    return {
        "total_alerts": total_alerts,
        "high_severity": high_severity,
        "distribution": dist_dict
    }
