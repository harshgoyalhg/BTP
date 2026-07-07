import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

USE_SQLITE = False

if USE_SQLITE:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./soc.db"
    # SQLite requires check_same_thread=False for FastAPI
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("POSTGRES_USER", "soc_admin")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "securepassword123")
    DB_NAME = os.getenv("POSTGRES_DB", "soc_db")
    DB_PORT = os.getenv("DB_PORT", "5432")
    SQLALCHEMY_DATABASE_URL = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
