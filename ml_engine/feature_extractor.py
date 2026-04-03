"""
URL Feature Extractor
Extracts 41 engineered features from URLs for the DNN branch
"""

import re
import math
import string
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Tuple
import numpy as np

# Import config for suspicious TLDs and URL shorteners
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import SUSPICIOUS_TLDS, URL_SHORTENERS
except ImportError:
    SUSPICIOUS_TLDS = {".xyz", ".top", ".buzz", ".club", ".tk", ".ml", ".ga"}
    URL_SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co"}


class FeatureExtractor:
    """
    Extracts 41 engineered features from URLs across 6 categories:
    1. URL Lexical Features (12)
    2. Domain Intelligence Features (8)
    3. Security Features (6)
    4. Content Features (7)
    5. Behavioral Features (5)
    6. Reputation Features (3)
    """
    
    # Feature names for interpretability
    FEATURE_NAMES = [
        # Lexical (0-11)
        "url_length", "num_dots", "num_hyphens", "num_underscores",
        "num_digits", "num_special_chars", "has_at_symbol", "path_depth",
        "num_query_params", "has_fragment", "url_entropy", "consecutive_consonants",
        
        # Domain Intel (12-19)
        "domain_length", "subdomain_depth", "is_ip_address", "uses_url_shortener",
        "suspicious_tld", "has_punycode", "domain_digit_ratio", "tld_length",
        
        # Security (20-25)
        "is_https", "has_port", "port_is_standard", "double_slash_in_path",
        "hex_encoded_chars", "suspicious_file_extension",
        
        # Content indicators (26-32)
        "has_login_keyword", "has_secure_keyword", "has_account_keyword",
        "has_update_keyword", "has_verify_keyword", "has_bank_keyword",
        "brand_impersonation_score",
        
        # Behavioral (33-37)
        "url_shortening_chain", "excessive_subdomains", "random_looking_domain",
        "long_subdomain", "path_length_ratio",
        
        # Reputation (38-40)
        "known_brand_in_subdomain", "misleading_tld", "homoglyph_score",
    ]
    
    # Common brand names for impersonation detection
    COMMON_BRANDS = {
        "paypal", "apple", "google", "microsoft", "amazon", "facebook",
        "netflix", "instagram", "twitter", "linkedin", "dropbox", "adobe",
        "spotify", "walmart", "ebay", "chase", "wellsfargo", "bankofamerica",
        "citibank", "usbank", "capitalone", "americanexpress", "discover",
    }
    
    # Homoglyph mappings (characters that look similar)
    HOMOGLYPHS = {
        'o': '0', '0': 'o', 'l': '1', '1': 'l', 'i': '1',
        'a': '4', '4': 'a', 'e': '3', '3': 'e', 's': '5',
        '5': 's', 'b': '8', '8': 'b', 'g': '9', '9': 'g',
    }
    
    def __init__(self):
        self.num_features = 41
        self.feature_names = self.FEATURE_NAMES
    
    def extract(self, url: str) -> np.ndarray:
        """
        Extract all 41 features from a URL.
        
        Args:
            url: URL string to analyze
            
        Returns:
            numpy array of shape (41,) with feature values
        """
        features = []
        
        # Clean URL - remove any control characters
        url = ''.join(c for c in str(url) if c.isprintable() or c in ' \t')
        
        # Parse URL with error handling
        try:
            parsed = urlparse(url.lower() if not url.startswith('http') else url)
            if not parsed.scheme:
                parsed = urlparse('http://' + url)
        except Exception:
            parsed = urlparse('http://invalid.url')
        
        domain = parsed.netloc.lower() if parsed.netloc else ''
        path = parsed.path if parsed.path else ''
        query = parsed.query if parsed.query else ''
        
        # Safe port access
        try:
            port = parsed.port
        except (ValueError, TypeError):
            port = None
        
        # === 1. Lexical Features (12) ===
        features.append(self._normalize(len(url), 0, 500))  # url_length
        features.append(self._normalize(url.count('.'), 0, 20))  # num_dots
        features.append(self._normalize(url.count('-'), 0, 20))  # num_hyphens
        features.append(self._normalize(url.count('_'), 0, 20))  # num_underscores
        features.append(self._normalize(sum(c.isdigit() for c in url), 0, 50))  # num_digits
        features.append(self._normalize(sum(c in '!@#$%^&*()+=[]{}|;:,<>?' for c in url), 0, 20))  # special_chars
        features.append(1.0 if '@' in url else 0.0)  # has_at_symbol
        features.append(self._normalize(path.count('/'), 0, 15))  # path_depth
        try:
            features.append(self._normalize(len(parse_qs(query)), 0, 20))  # num_query_params
        except Exception:
            features.append(0.0)
        features.append(1.0 if parsed.fragment else 0.0)  # has_fragment
        features.append(self._calculate_entropy(url))  # url_entropy
        features.append(self._consecutive_consonants(url))  # consecutive_consonants
        
        # === 2. Domain Intelligence Features (8) ===
        features.append(self._normalize(len(domain), 0, 100))  # domain_length
        features.append(self._normalize(domain.count('.'), 0, 10))  # subdomain_depth
        features.append(1.0 if self._is_ip_address(domain) else 0.0)  # is_ip_address
        features.append(1.0 if self._uses_url_shortener(domain) else 0.0)  # uses_url_shortener
        features.append(1.0 if self._has_suspicious_tld(domain) else 0.0)  # suspicious_tld
        features.append(1.0 if domain.startswith('xn--') or 'xn--' in domain else 0.0)  # has_punycode
        features.append(self._digit_ratio(domain))  # domain_digit_ratio
        features.append(self._normalize(len(self._get_tld(domain)), 0, 10))  # tld_length
        
        # === 3. Security Features (6) ===
        features.append(1.0 if parsed.scheme == 'https' else 0.0)  # is_https
        features.append(1.0 if ':' in domain and self._has_port(domain) else 0.0)  # has_port
        features.append(1.0 if port in [None, 80, 443] else 0.0)  # port_is_standard
        features.append(1.0 if '//' in path else 0.0)  # double_slash_in_path
        features.append(self._normalize(url.count('%'), 0, 20))  # hex_encoded_chars
        features.append(1.0 if self._has_suspicious_extension(path) else 0.0)  # suspicious_extension
        
        # === 4. Content Indicator Features (7) ===
        url_lower = url.lower()
        features.append(1.0 if any(kw in url_lower for kw in ['login', 'signin', 'log-in', 'sign-in']) else 0.0)
        features.append(1.0 if any(kw in url_lower for kw in ['secure', 'security', 'ssl']) else 0.0)
        features.append(1.0 if any(kw in url_lower for kw in ['account', 'myaccount', 'profile']) else 0.0)
        features.append(1.0 if any(kw in url_lower for kw in ['update', 'upgrade', 'renew']) else 0.0)
        features.append(1.0 if any(kw in url_lower for kw in ['verify', 'confirm', 'validate']) else 0.0)
        features.append(1.0 if any(kw in url_lower for kw in ['bank', 'banking', 'financial']) else 0.0)
        features.append(self._brand_impersonation_score(domain))  # brand_impersonation
        
        # === 5. Behavioral Features (5) ===
        features.append(self._url_shortening_chain_score(url))  # shortening chain
        features.append(1.0 if domain.count('.') > 4 else 0.0)  # excessive_subdomains
        features.append(self._random_looking_score(domain))  # random_looking_domain
        features.append(1.0 if self._has_long_subdomain(domain) else 0.0)  # long_subdomain
        features.append(self._normalize(len(path), 0, 200) if len(url) > 0 else 0.0)  # path_length_ratio
        
        # === 6. Reputation Features (3) ===
        features.append(1.0 if self._brand_in_subdomain(domain) else 0.0)  # known_brand_in_subdomain
        features.append(self._misleading_tld_score(domain))  # misleading_tld
        features.append(self._homoglyph_score(domain))  # homoglyph_score
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch(self, urls: List[str]) -> np.ndarray:
        """Extract features for multiple URLs."""
        return np.array([self.extract(url) for url in urls], dtype=np.float32)
    
    # === Helper Methods ===
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range."""
        return min(1.0, max(0.0, (value - min_val) / (max_val - min_val + 1e-8)))
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        entropy = -sum(p * math.log2(p + 1e-10) for p in prob if p > 0)
        return self._normalize(entropy, 0, 6)  # Normalize to ~[0, 1]
    
    def _consecutive_consonants(self, url: str) -> float:
        """Count max consecutive consonants (indicates random strings)."""
        consonants = 'bcdfghjklmnpqrstvwxyz'
        max_count = 0
        current = 0
        for char in url.lower():
            if char in consonants:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return self._normalize(max_count, 0, 10)
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address."""
        # Remove port if present
        domain = domain.split(':')[0]
        parts = domain.split('.')
        if len(parts) == 4:
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                return False
        return False
    
    def _uses_url_shortener(self, domain: str) -> bool:
        """Check if domain is a known URL shortener."""
        return any(shortener in domain for shortener in URL_SHORTENERS)
    
    def _has_suspicious_tld(self, domain: str) -> bool:
        """Check for suspicious TLD."""
        for tld in SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return True
        return False
    
    def _get_tld(self, domain: str) -> str:
        """Extract TLD from domain."""
        parts = domain.split('.')
        return parts[-1] if parts else ''
    
    def _digit_ratio(self, domain: str) -> float:
        """Calculate ratio of digits in domain."""
        if not domain:
            return 0.0
        return sum(c.isdigit() for c in domain) / len(domain)
    
    def _has_port(self, domain: str) -> bool:
        """Check if domain has a port number."""
        if ':' in domain:
            try:
                port = int(domain.split(':')[-1])
                return 1 <= port <= 65535
            except ValueError:
                return False
        return False
    
    def _has_suspicious_extension(self, path: str) -> bool:
        """Check for suspicious file extensions."""
        suspicious = ['.exe', '.zip', '.rar', '.js', '.php', '.asp', '.scr', '.bat', '.cmd']
        return any(path.lower().endswith(ext) for ext in suspicious)
    
    def _brand_impersonation_score(self, domain: str) -> float:
        """Calculate brand impersonation score."""
        domain_lower = domain.lower()
        score = 0.0
        
        for brand in self.COMMON_BRANDS:
            if brand in domain_lower:
                # Check if it's not the actual brand domain
                actual_domains = [f"{brand}.com", f"{brand}.net", f"{brand}.org", f"www.{brand}.com"]
                if not any(domain_lower == actual or domain_lower.endswith('.' + actual.split('www.')[-1]) 
                          for actual in actual_domains):
                    score = max(score, 0.8)
                    
                # Check for typosquatting patterns
                if re.search(rf'{brand}\d', domain_lower) or re.search(rf'\d{brand}', domain_lower):
                    score = max(score, 0.95)
        
        return score
    
    def _url_shortening_chain_score(self, url: str) -> float:
        """Detect potential URL shortening chains."""
        shortener_count = sum(1 for s in URL_SHORTENERS if s in url.lower())
        return self._normalize(shortener_count, 0, 3)
    
    def _random_looking_score(self, domain: str) -> float:
        """Score how random/generated the domain looks."""
        # Remove TLD
        parts = domain.split('.')
        if len(parts) > 1:
            main_domain = parts[-2]
        else:
            main_domain = domain
            
        if not main_domain:
            return 0.0
            
        # Check for patterns indicating randomness
        score = 0.0
        
        # High consonant ratio
        consonants = sum(1 for c in main_domain if c in 'bcdfghjklmnpqrstvwxyz')
        if len(main_domain) > 0 and consonants / len(main_domain) > 0.7:
            score += 0.3
            
        # Mix of numbers and letters
        has_digit = any(c.isdigit() for c in main_domain)
        has_letter = any(c.isalpha() for c in main_domain)
        if has_digit and has_letter:
            score += 0.3
            
        # Long domain without real words
        if len(main_domain) > 15:
            score += 0.4
            
        return min(1.0, score)
    
    def _has_long_subdomain(self, domain: str) -> bool:
        """Check for unusually long subdomains."""
        parts = domain.split('.')
        if len(parts) > 2:
            subdomains = parts[:-2]
            return any(len(sub) > 20 for sub in subdomains)
        return False
    
    def _brand_in_subdomain(self, domain: str) -> bool:
        """Check if a known brand appears in subdomain (suspicious)."""
        parts = domain.split('.')
        if len(parts) > 2:
            subdomains = '.'.join(parts[:-2]).lower()
            return any(brand in subdomains for brand in self.COMMON_BRANDS)
        return False
    
    def _misleading_tld_score(self, domain: str) -> float:
        """Score for TLDs that might be misleading."""
        misleading_patterns = {
            '.com-': 0.9,  # com-something.xyz
            '-com.': 0.9,
            '.org-': 0.8,
            '-secure': 0.7,
            '-login': 0.8,
            '-verify': 0.8,
        }
        
        for pattern, score in misleading_patterns.items():
            if pattern in domain.lower():
                return score
        return 0.0
    
    def _homoglyph_score(self, domain: str) -> float:
        """Detect potential homoglyph attacks."""
        domain_lower = domain.lower()
        score = 0.0
        
        for brand in self.COMMON_BRANDS:
            # Check for homoglyph substitutions
            for orig, sub in self.HOMOGLYPHS.items():
                modified_brand = brand.replace(orig, sub)
                if modified_brand != brand and modified_brand in domain_lower:
                    score = max(score, 0.9)
                    break
        
        return score


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://paypa1-secure-login.xyz/verify",
        "http://192.168.1.1/admin/login.php",
        "http://bit.ly/a3xK9f",
        "https://microsoft-account-verify-secure.tk/update",
    ]
    
    print(f"Number of features: {extractor.num_features}")
    print()
    
    for url in test_urls:
        features = extractor.extract(url)
        print(f"URL: {url}")
        print(f"Features shape: {features.shape}")
        print(f"Non-zero features: {np.sum(features > 0)}")
        
        # Show top features
        top_indices = np.argsort(features)[-5:][::-1]
        print("Top 5 features:")
        for idx in top_indices:
            if features[idx] > 0:
                print(f"  {extractor.feature_names[idx]}: {features[idx]:.3f}")
        print()
