"""
Pydantic Models for API Request/Response Schemas
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class ThreatCategory(str, Enum):
    """Threat category enumeration."""
    SAFE = "safe"
    PHISHING = "phishing"
    MALWARE = "malware"
    DATA_LEAK = "data_leak"
    SCAM = "scam"


class ThreatLevel(str, Enum):
    """Threat severity level."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# === Request Models ===

class URLScanRequest(BaseModel):
    """Request model for URL scanning."""
    url: str = Field(..., description="URL to analyze", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.example.com/login"
            }
        }


class BatchScanRequest(BaseModel):
    """Request model for batch URL scanning."""
    urls: List[str] = Field(..., description="List of URLs to analyze", min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://www.google.com",
                    "http://suspicious-site.xyz/login"
                ]
            }
        }


class BlockSiteRequest(BaseModel):
    """Request for blocking a site."""
    url: str = Field(..., description="URL or domain to block")
    reason: Optional[str] = Field(None, description="Reason for blocking")
    category: Optional[ThreatCategory] = Field(None, description="Threat category")


class UnblockSiteRequest(BaseModel):
    """Request for unblocking a site."""
    url: str = Field(..., description="URL or domain to unblock")
    reason: Optional[str] = Field(None, description="Reason for unblocking")


class SettingsUpdateRequest(BaseModel):
    """Request for updating settings."""
    ml_protection_enabled: Optional[bool] = Field(None, description="Enable/disable ML protection")
    sensitivity_level: Optional[str] = Field(None, description="Detection sensitivity: low, medium, high")
    auto_block_threats: Optional[bool] = Field(None, description="Automatically block detected threats")
    notification_enabled: Optional[bool] = Field(None, description="Enable threat notifications")


# === Response Models ===

class RiskFactor(BaseModel):
    """Individual risk factor in threat explanation."""
    factor: str = Field(..., description="Human-readable risk description")
    severity: str = Field(..., description="Severity level: high, medium, low")
    icon: str = Field(..., description="Emoji icon for the risk")


class ThreatProbabilities(BaseModel):
    """Probability distribution over threat classes."""
    safe: float = Field(..., ge=0, le=1)
    phishing: float = Field(..., ge=0, le=1)
    malware: float = Field(..., ge=0, le=1)
    data_leak: float = Field(..., ge=0, le=1)
    scam: float = Field(..., ge=0, le=1)


class ScanResult(BaseModel):
    """Complete scan result for a single URL."""
    url: str = Field(..., description="The scanned URL")
    threat_level: ThreatLevel = Field(..., description="Overall threat level")
    category: ThreatCategory = Field(..., description="Detected threat category")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    icon: str = Field(..., description="Status icon")
    color: str = Field(..., description="UI color indicator")
    reasons: List[RiskFactor] = Field(..., description="List of risk factors")
    recommendation: str = Field(..., description="Action recommendation")
    blocked: bool = Field(..., description="Whether the URL should be blocked")
    probabilities: ThreatProbabilities = Field(..., description="Class probabilities")
    scan_time_ms: float = Field(..., description="Scan processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://paypa1-secure.xyz/login",
                "threat_level": "CRITICAL",
                "category": "phishing",
                "confidence": 0.94,
                "risk_score": 92,
                "icon": "🔴",
                "color": "red",
                "reasons": [
                    {"factor": "Uses suspicious TLD (.xyz)", "severity": "high", "icon": "🔴"},
                    {"factor": "Appears to impersonate a known brand", "severity": "high", "icon": "🔴"}
                ],
                "recommendation": "DO NOT enter any personal information.",
                "blocked": True,
                "probabilities": {
                    "safe": 0.02,
                    "phishing": 0.94,
                    "malware": 0.02,
                    "data_leak": 0.01,
                    "scam": 0.01
                },
                "scan_time_ms": 45.2
            }
        }


class BatchScanResult(BaseModel):
    """Results for batch URL scanning."""
    total: int = Field(..., description="Total URLs scanned")
    threats_found: int = Field(..., description="Number of threats detected")
    results: List[ScanResult] = Field(..., description="Individual scan results")
    total_time_ms: float = Field(..., description="Total processing time")


class BlockedSite(BaseModel):
    """Blocked site record."""
    id: int = Field(..., description="Record ID")
    url: str = Field(..., description="Blocked URL/domain")
    category: ThreatCategory = Field(..., description="Threat category")
    reason: Optional[str] = Field(None, description="Block reason")
    blocked_at: datetime = Field(..., description="When the site was blocked")
    auto_blocked: bool = Field(..., description="Whether auto-blocked by ML")


class BlockedSitesList(BaseModel):
    """Paginated list of blocked sites."""
    items: List[BlockedSite] = Field(..., description="Blocked sites")
    total: int = Field(..., description="Total count")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Items per page")


class ScanHistoryItem(BaseModel):
    """Single scan history record."""
    id: int
    url: str
    category: ThreatCategory
    threat_level: ThreatLevel
    confidence: float
    risk_score: int
    blocked: bool
    scanned_at: datetime


class ScanHistory(BaseModel):
    """Paginated scan history."""
    items: List[ScanHistoryItem]
    total: int
    page: int
    per_page: int


class AnalyticsData(BaseModel):
    """Threat analytics data."""
    total_scans: int = Field(..., description="Total scans performed")
    threats_blocked: int = Field(..., description="Total threats blocked")
    safe_urls: int = Field(..., description="Safe URLs scanned")
    by_category: Dict[str, int] = Field(..., description="Counts by threat category")
    recent_threats: List[ScanHistoryItem] = Field(..., description="Recent threat detections")
    daily_stats: List[Dict] = Field(..., description="Daily scan statistics")


class Settings(BaseModel):
    """Application settings."""
    ml_protection_enabled: bool = Field(True, description="ML protection status")
    sensitivity_level: str = Field("medium", description="Detection sensitivity")
    auto_block_threats: bool = Field(True, description="Auto-block threats")
    notification_enabled: bool = Field(True, description="Notifications enabled")
    last_updated: datetime = Field(..., description="Last settings update")


class HealthCheck(BaseModel):
    """API health check response."""
    status: str = Field("healthy", description="API status")
    model_loaded: bool = Field(..., description="ML model loaded status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Server uptime")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional details")
