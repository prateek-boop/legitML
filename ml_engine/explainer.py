"""
Threat Explainer Module
Provides human-readable explanations for threat predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.feature_extractor import FeatureExtractor

try:
    from config import THREAT_CLASSES, THREAT_LEVELS
except ImportError:
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]
    THREAT_LEVELS = {
        "safe": {"level": "SAFE", "color": "green", "icon": "✅"},
        "phishing": {"level": "CRITICAL", "color": "red", "icon": "🔴"},
        "malware": {"level": "CRITICAL", "color": "red", "icon": "🔴"},
        "data_leak": {"level": "HIGH", "color": "orange", "icon": "🟠"},
        "scam": {"level": "HIGH", "color": "orange", "icon": "🟠"},
    }


@dataclass
class RiskFactor:
    """Represents a single risk factor in the explanation."""
    factor: str
    severity: str  # "high", "medium", "low"
    icon: str
    feature_name: str
    feature_value: float


class ThreatExplainer:
    """
    Generates human-readable explanations for threat predictions.
    
    Maps feature values to understandable risk factors and produces
    structured threat reports.
    """
    
    # Feature to explanation templates
    FEATURE_EXPLANATIONS = {
        # Lexical features
        "url_length": {
            "threshold": 0.6,
            "template": "URL is unusually long ({value:.0%} of max)",
            "severity": "medium"
        },
        "num_dots": {
            "threshold": 0.3,
            "template": "URL contains many dots/subdomains",
            "severity": "medium"
        },
        "num_hyphens": {
            "threshold": 0.3,
            "template": "URL contains excessive hyphens (often used in spoofing)",
            "severity": "medium"
        },
        "has_at_symbol": {
            "threshold": 0.5,
            "template": "URL contains @ symbol (can be used to deceive)",
            "severity": "high"
        },
        "url_entropy": {
            "threshold": 0.7,
            "template": "URL appears randomly generated",
            "severity": "high"
        },
        
        # Domain features
        "is_ip_address": {
            "threshold": 0.5,
            "template": "Uses IP address instead of domain name",
            "severity": "high"
        },
        "uses_url_shortener": {
            "threshold": 0.5,
            "template": "Uses URL shortener to hide destination",
            "severity": "high"
        },
        "suspicious_tld": {
            "threshold": 0.5,
            "template": "Uses suspicious top-level domain (.xyz, .tk, etc.)",
            "severity": "high"
        },
        "has_punycode": {
            "threshold": 0.5,
            "template": "Uses internationalized characters (potential homograph attack)",
            "severity": "high"
        },
        "domain_digit_ratio": {
            "threshold": 0.3,
            "template": "Domain contains unusual number of digits",
            "severity": "medium"
        },
        
        # Security features
        "is_https": {
            "threshold": 0.5,
            "invert": True,
            "template": "Does not use HTTPS encryption",
            "severity": "high"
        },
        "has_port": {
            "threshold": 0.5,
            "template": "Uses non-standard port number",
            "severity": "medium"
        },
        "double_slash_in_path": {
            "threshold": 0.5,
            "template": "Contains suspicious double slashes in path",
            "severity": "medium"
        },
        "hex_encoded_chars": {
            "threshold": 0.2,
            "template": "Contains encoded characters (potential obfuscation)",
            "severity": "medium"
        },
        "suspicious_file_extension": {
            "threshold": 0.5,
            "template": "Links to executable or archive file",
            "severity": "high"
        },
        
        # Content indicators
        "has_login_keyword": {
            "threshold": 0.5,
            "template": "Contains login/signin keywords",
            "severity": "medium"
        },
        "has_verify_keyword": {
            "threshold": 0.5,
            "template": "Contains verification/confirmation keywords",
            "severity": "medium"
        },
        "has_bank_keyword": {
            "threshold": 0.5,
            "template": "Contains banking/financial keywords",
            "severity": "medium"
        },
        "brand_impersonation_score": {
            "threshold": 0.5,
            "template": "Appears to impersonate a known brand",
            "severity": "high"
        },
        
        # Behavioral features
        "excessive_subdomains": {
            "threshold": 0.5,
            "template": "Uses excessive subdomain nesting",
            "severity": "high"
        },
        "random_looking_domain": {
            "threshold": 0.5,
            "template": "Domain appears randomly generated",
            "severity": "high"
        },
        "long_subdomain": {
            "threshold": 0.5,
            "template": "Contains unusually long subdomain",
            "severity": "medium"
        },
        
        # Reputation features
        "known_brand_in_subdomain": {
            "threshold": 0.5,
            "template": "Known brand name appears in subdomain (suspicious)",
            "severity": "high"
        },
        "misleading_tld": {
            "threshold": 0.5,
            "template": "Uses misleading TLD pattern (e.g., .com-verify)",
            "severity": "high"
        },
        "homoglyph_score": {
            "threshold": 0.5,
            "template": "Uses look-alike characters (homoglyph attack)",
            "severity": "high"
        },
    }
    
    # Recommendations by threat class
    RECOMMENDATIONS = {
        "safe": "This URL appears to be safe. However, always verify before entering sensitive information.",
        "phishing": "DO NOT enter any personal information. This site is designed to steal your credentials.",
        "malware": "DO NOT download any files. This site may contain malicious software.",
        "data_leak": "This site may collect and misuse your personal data. Proceed with extreme caution.",
        "scam": "This appears to be a scam. Do not engage or provide any information.",
    }
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.FEATURE_NAMES
    
    def explain(
        self,
        url: str,
        prediction: Dict,
        features: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Generate a comprehensive threat explanation.
        
        Args:
            url: The analyzed URL
            prediction: Model prediction dict with class, confidence, probabilities
            features: Pre-computed features (optional, will extract if not provided)
            top_k: Number of top risk factors to include
            
        Returns:
            Structured explanation dictionary
        """
        # Extract features if not provided
        if features is None:
            features = self.feature_extractor.extract(url)
        
        threat_class = prediction["class"]
        confidence = prediction["confidence"]
        
        # Get risk factors
        risk_factors = self._get_risk_factors(features, threat_class)
        
        # Sort by severity and limit
        severity_order = {"high": 0, "medium": 1, "low": 2}
        risk_factors.sort(key=lambda x: (severity_order[x.severity], -x.feature_value))
        top_factors = risk_factors[:top_k]
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(threat_class, confidence, features)
        
        # Build explanation
        explanation = {
            "url": url,
            "threat_level": THREAT_LEVELS[threat_class]["level"],
            "category": threat_class,
            "confidence": confidence,
            "risk_score": risk_score,
            "icon": THREAT_LEVELS[threat_class]["icon"],
            "color": THREAT_LEVELS[threat_class]["color"],
            "reasons": [
                {
                    "factor": rf.factor,
                    "severity": rf.severity,
                    "icon": rf.icon,
                }
                for rf in top_factors
            ],
            "recommendation": self.RECOMMENDATIONS[threat_class],
            "blocked": threat_class != "safe",
            "all_probabilities": prediction.get("probabilities", {}),
        }
        
        return explanation
    
    def _get_risk_factors(
        self, 
        features: np.ndarray, 
        threat_class: str
    ) -> List[RiskFactor]:
        """Extract risk factors from feature values."""
        factors = []
        
        for i, (name, value) in enumerate(zip(self.feature_names, features)):
            if name not in self.FEATURE_EXPLANATIONS:
                continue
            
            config = self.FEATURE_EXPLANATIONS[name]
            threshold = config["threshold"]
            
            # Handle inverted features (like is_https where low = bad)
            check_value = 1 - value if config.get("invert", False) else value
            
            if check_value >= threshold:
                severity = config["severity"]
                icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                
                template = config["template"]
                if "{value" in template:
                    factor_text = template.format(value=value)
                else:
                    factor_text = template
                
                factors.append(RiskFactor(
                    factor=factor_text,
                    severity=severity,
                    icon=icon,
                    feature_name=name,
                    feature_value=float(value)
                ))
        
        return factors
    
    def _calculate_risk_score(
        self, 
        threat_class: str, 
        confidence: float,
        features: np.ndarray
    ) -> int:
        """Calculate overall risk score (0-100)."""
        if threat_class == "safe":
            # Even safe URLs might have some risk
            base_score = 10
        else:
            # Base score by threat type
            base_scores = {
                "phishing": 80,
                "malware": 90,
                "data_leak": 70,
                "scam": 75,
            }
            base_score = base_scores.get(threat_class, 50)
        
        # Adjust by confidence
        score = base_score * confidence
        
        # Add feature-based adjustments
        high_risk_features = [
            "is_ip_address", "brand_impersonation_score", "homoglyph_score",
            "suspicious_tld", "suspicious_file_extension"
        ]
        
        for name in high_risk_features:
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                score += features[idx] * 5
        
        return min(100, max(0, int(score)))
    
    def explain_batch(
        self,
        urls: List[str],
        predictions: List[Dict],
        features: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Generate explanations for multiple URLs."""
        if features is None:
            features = self.feature_extractor.extract_batch(urls)
        
        return [
            self.explain(url, pred, feat)
            for url, pred, feat in zip(urls, predictions, features)
        ]


if __name__ == "__main__":
    # Test the explainer
    explainer = ThreatExplainer()
    
    test_cases = [
        ("https://www.google.com/search?q=test", "safe", 0.95),
        ("http://paypa1-secure-login.xyz/verify", "phishing", 0.94),
        ("http://free-software-crack.tk/download.exe", "malware", 0.88),
        ("http://claim-your-prize-now.top/winner", "scam", 0.82),
    ]
    
    print("=" * 60)
    print("THREAT EXPLAINER TEST")
    print("=" * 60)
    
    for url, threat_class, confidence in test_cases:
        prediction = {
            "class": threat_class,
            "class_index": THREAT_CLASSES.index(threat_class),
            "confidence": confidence,
        }
        
        explanation = explainer.explain(url, prediction)
        
        print(f"\n{explanation['icon']} URL: {url}")
        print(f"   Threat Level: {explanation['threat_level']}")
        print(f"   Category: {explanation['category']}")
        print(f"   Confidence: {explanation['confidence']:.0%}")
        print(f"   Risk Score: {explanation['risk_score']}/100")
        print(f"   Blocked: {explanation['blocked']}")
        print("   Risk Factors:")
        for reason in explanation['reasons']:
            print(f"      {reason['icon']} {reason['factor']}")
        print(f"   Recommendation: {explanation['recommendation']}")
