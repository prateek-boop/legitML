"""
URL Scanning API Routes
"""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.schemas import (
    URLScanRequest, BatchScanRequest,
    ScanResult, BatchScanResult, ThreatProbabilities, RiskFactor,
    ThreatCategory, ThreatLevel
)

router = APIRouter(prefix="/api/v1/scan", tags=["Scanning"])


# Global model instance (loaded by server on startup)
_model = None
_tokenizer = None
_extractor = None
_explainer = None


def get_model():
    """Dependency to get loaded model."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be starting."
        )
    return _model


def init_model(model, tokenizer, extractor, explainer):
    """Initialize model components (called by server on startup)."""
    global _model, _tokenizer, _extractor, _explainer
    _model = model
    _tokenizer = tokenizer
    _extractor = extractor
    _explainer = explainer


@router.post("", response_model=ScanResult)
@router.post("/", response_model=ScanResult)
async def scan_url(request: URLScanRequest, model=Depends(get_model)):
    """
    Analyze a URL for threats.
    
    Returns detailed threat analysis including:
    - Threat category (safe, phishing, malware, data_leak, scam)
    - Confidence score
    - Risk factors with explanations
    - Recommended action
    """
    start_time = time.time()
    
    url = request.url.strip()
    
    # Tokenize and extract features
    url_tokens = _tokenizer.tokenize_batch([url])
    features = _extractor.extract_batch([url])
    
    # Get prediction
    predictions = model.predict_with_confidence(url_tokens, features)
    prediction = predictions[0]
    
    # Generate explanation
    explanation = _explainer.explain(url, prediction, features[0])
    
    # Calculate scan time
    scan_time_ms = (time.time() - start_time) * 1000
    
    # Build response
    return ScanResult(
        url=url,
        threat_level=ThreatLevel(explanation["threat_level"]),
        category=ThreatCategory(explanation["category"]),
        confidence=explanation["confidence"],
        risk_score=explanation["risk_score"],
        icon=explanation["icon"],
        color=explanation["color"],
        reasons=[
            RiskFactor(**r) for r in explanation["reasons"]
        ],
        recommendation=explanation["recommendation"],
        blocked=explanation["blocked"],
        probabilities=ThreatProbabilities(**prediction["probabilities"]),
        scan_time_ms=round(scan_time_ms, 2)
    )


@router.post("/batch", response_model=BatchScanResult)
async def scan_urls_batch(request: BatchScanRequest, model=Depends(get_model)):
    """
    Analyze multiple URLs for threats.
    
    Processes up to 100 URLs in a single request for efficiency.
    """
    start_time = time.time()
    
    urls = [url.strip() for url in request.urls]
    
    # Batch process
    url_tokens = _tokenizer.tokenize_batch(urls)
    features = _extractor.extract_batch(urls)
    
    # Get predictions
    predictions = model.predict_with_confidence(url_tokens, features)
    
    # Generate explanations
    results = []
    threats_found = 0
    
    for i, (url, pred, feat) in enumerate(zip(urls, predictions, features)):
        explanation = _explainer.explain(url, pred, feat)
        
        if explanation["blocked"]:
            threats_found += 1
        
        results.append(ScanResult(
            url=url,
            threat_level=ThreatLevel(explanation["threat_level"]),
            category=ThreatCategory(explanation["category"]),
            confidence=explanation["confidence"],
            risk_score=explanation["risk_score"],
            icon=explanation["icon"],
            color=explanation["color"],
            reasons=[RiskFactor(**r) for r in explanation["reasons"]],
            recommendation=explanation["recommendation"],
            blocked=explanation["blocked"],
            probabilities=ThreatProbabilities(**pred["probabilities"]),
            scan_time_ms=0  # Individual times not tracked in batch
        ))
    
    total_time_ms = (time.time() - start_time) * 1000
    
    return BatchScanResult(
        total=len(urls),
        threats_found=threats_found,
        results=results,
        total_time_ms=round(total_time_ms, 2)
    )
