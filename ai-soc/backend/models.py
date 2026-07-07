from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from .database import Base

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    attack = Column(String(100), index=True)
    confidence = Column(Float)
    severity = Column(String(50))

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    attack = Column(String(100), index=True)
    investigation = Column(Text)
    response = Column(Text)
    report = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AgentLog(Base):
    __tablename__ = "agent_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100))
    action = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
