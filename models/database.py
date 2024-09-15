import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class FeatureImportanceTask(Base):
    __tablename__ = "feature_importance_tasks"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    error = Column(String, nullable=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    disabled = Column(Boolean, default=False)