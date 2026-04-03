"""
ShieldNet Configuration
Central configuration for the ML-powered threat detection system
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
ML_ENGINE_DIR = BASE_DIR / "ml_engine"
SAVED_MODEL_DIR = ML_ENGINE_DIR / "saved_model"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    "url_max_length": 200,           # Max URL characters for CNN input
    "char_vocab_size": 74,           # Character vocabulary size
    "embedding_dim": 32,             # Character embedding dimension
    "cnn_filters": 128,              # CNN filter count
    "dnn_units": [256, 128, 64],     # DNN layer sizes
    "num_features": 41,              # Number of engineered features
    "num_classes": 5,                # Output classes
    "dropout_rate": 0.4,             # Dropout rate
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.001,
    "early_stopping_patience": 7,  # Less patience to avoid overfitting
    "validation_split": 0.1,
    "test_split": 0.1,
}

# Threat Classes
THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]

# Class Weights for Imbalanced Data
CLASS_WEIGHTS = {
    0: 1.5,    # safe (increased to reduce False Positives)
    1: 1.0,    # phishing
    2: 1.2,    # malware
    3: 1.2,    # data_leak
    4: 1.5,    # scam
}

# Threat Levels
THREAT_LEVELS = {
    "safe": {"level": "SAFE", "color": "green", "icon": "✅"},
    "phishing": {"level": "CRITICAL", "color": "red", "icon": "🔴"},
    "malware": {"level": "CRITICAL", "color": "red", "icon": "🔴"},
    "data_leak": {"level": "HIGH", "color": "orange", "icon": "🟠"},
    "scam": {"level": "HIGH", "color": "orange", "icon": "🟠"},
}

# Suspicious TLDs
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".buzz", ".club", ".work", ".click", ".link",
    ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc", ".ws",
}

# URL Shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "buff.ly",
    "ow.ly", "shorturl.at", "cutt.ly", "rb.gy", "short.io",
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "cors_origins": ["*"],
}

# Database Configuration
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR / 'shieldnet.db'}"
