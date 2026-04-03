"""
FastAPI Production Server
ShieldNet Threat Detection API
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routes
from api.routes.scan import router as scan_router, init_model
from api.models.schemas import HealthCheck, ErrorResponse

# Import ML components
from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor
from ml_engine.model import ThreatDetectionModel
from ml_engine.explainer import ThreatExplainer

try:
    from config import API_CONFIG, SAVED_MODEL_DIR
except ImportError:
    API_CONFIG = {"host": "0.0.0.0", "port": 8000, "cors_origins": ["*"]}
    SAVED_MODEL_DIR = Path(__file__).parent.parent / "ml_engine" / "saved_model"

# Global state
_start_time = time.time()
_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - load model on startup."""
    global _model_loaded
    
    print("\n🛡️ ShieldNet API Starting...")
    print("=" * 50)
    
    # Initialize components
    print("📦 Loading ML components...")
    tokenizer = URLTokenizer()
    extractor = FeatureExtractor()
    explainer = ThreatExplainer()
    
    # Load or create model
    model = ThreatDetectionModel()
    model_path = SAVED_MODEL_DIR / "shieldnet_model.keras"
    
    if model_path.exists():
        print(f"📂 Loading trained model from {model_path}")
        model.load(str(model_path))
    else:
        print("⚠️ No trained model found. Using untrained model.")
        print("   Run 'python ml_engine/train_model.py' to train the model.")
    
    # Initialize route dependencies
    init_model(model, tokenizer, extractor, explainer)
    _model_loaded = True
    
    print("✅ Model loaded successfully!")
    print("=" * 50)
    print(f"🚀 Server ready at http://localhost:{API_CONFIG.get('port', 8000)}")
    print(f"📖 API docs at http://localhost:{API_CONFIG.get('port', 8000)}/docs")
    print()
    
    yield
    
    # Cleanup
    print("\n🛑 ShieldNet API Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="ShieldNet API",
    description="""
    🛡️ **ShieldNet** - ML-Powered Malicious Website Detection API
    
    Analyze URLs for threats using deep learning:
    - **Phishing** detection (brand impersonation, credential harvesting)
    - **Malware** distribution sites
    - **Data leak** risks (privacy violations)
    - **Scam** detection (too-good-to-be-true offers)
    
    ## Quick Start
    ```bash
    curl -X POST http://localhost:8000/api/v1/scan \\
      -H "Content-Type: application/json" \\
      -d '{"url": "http://suspicious-site.xyz/login"}'
    ```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__}
        ).model_dump()
    )


# Include routers
app.include_router(scan_router)


# Root endpoints
@app.get("/", tags=["Root"])
async def root():
    """API root - welcome message."""
    return {
        "name": "ShieldNet API",
        "version": "1.0.0",
        "description": "ML-Powered Malicious Website Detection",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthCheck(
        status="healthy",
        model_loaded=_model_loaded,
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 2)
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=True
    )
