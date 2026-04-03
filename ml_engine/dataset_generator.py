"""
Synthetic Dataset Generator for Training
Generates realistic labeled URL samples across 5 threat categories
"""

import random
import string
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor

try:
    from config import THREAT_CLASSES, SUSPICIOUS_TLDS, URL_SHORTENERS, SAVED_MODEL_DIR, DATA_DIR
except ImportError:
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]
    SUSPICIOUS_TLDS = {".xyz", ".top", ".buzz", ".club", ".tk"}
    URL_SHORTENERS = {"bit.ly", "tinyurl.com"}
    DATA_DIR = Path(__file__).parent.parent / "data"


class DatasetGenerator:
    """
    Generates synthetic URLs with realistic feature correlations.
    
    Distribution:
    - Safe (40%): Legitimate-looking URLs from known patterns
    - Phishing (25%): Brand impersonation, suspicious keywords
    - Malware (15%): Executable downloads, obfuscated paths
    - Data Leak (12%): No HTTPS, excessive forms
    - Scam (8%): Too-good-to-be-true, urgency keywords
    """
    
    # Legitimate domains for safe URLs
    SAFE_DOMAINS = [
        "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
        "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
        "microsoft.com", "apple.com", "github.com", "stackoverflow.com", "medium.com",
        "nytimes.com", "bbc.com", "cnn.com", "spotify.com", "dropbox.com",
        "salesforce.com", "adobe.com", "zoom.us", "slack.com", "notion.so",
    ]
    
    # Brands commonly impersonated in phishing
    PHISHING_BRANDS = [
        "paypal", "apple", "google", "microsoft", "amazon", "facebook",
        "netflix", "instagram", "chase", "wellsfargo", "bankofamerica",
        "citibank", "usbank", "capitalone", "americanexpress",
    ]
    
    # Phishing keywords
    PHISHING_KEYWORDS = [
        "login", "signin", "verify", "secure", "account", "update",
        "confirm", "validate", "authenticate", "password", "credential",
    ]
    
    # Malware indicators
    MALWARE_EXTENSIONS = [".exe", ".zip", ".rar", ".msi", ".bat", ".scr", ".dll"]
    MALWARE_PATHS = [
        "/download/", "/free/", "/crack/", "/keygen/", "/patch/",
        "/update/", "/setup/", "/install/", "/driver/",
    ]
    
    # Scam keywords
    SCAM_KEYWORDS = [
        "winner", "prize", "free", "congratulations", "claim", "urgent",
        "limited", "exclusive", "reward", "bonus", "jackpot", "lottery",
    ]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.tokenizer = URLTokenizer()
        self.extractor = FeatureExtractor()
    
    def generate_dataset(
        self, 
        n_samples: int = 50000,
        distribution: Dict[str, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generate a complete labeled dataset.
        
        Args:
            n_samples: Total number of samples
            distribution: Dict mapping class names to proportions
            
        Returns:
            url_tokens: Shape (n_samples, 200)
            features: Shape (n_samples, 41)
            labels: Shape (n_samples,)
            urls: List of generated URL strings
        """
        distribution = distribution or {
            "safe": 0.40,
            "phishing": 0.25,
            "malware": 0.15,
            "data_leak": 0.12,
            "scam": 0.08,
        }
        
        urls = []
        labels = []
        
        for class_name, proportion in distribution.items():
            class_idx = THREAT_CLASSES.index(class_name)
            n_class = int(n_samples * proportion)
            
            print(f"Generating {n_class} {class_name} samples...")
            
            for _ in range(n_class):
                if class_name == "safe":
                    url = self._generate_safe_url()
                elif class_name == "phishing":
                    url = self._generate_phishing_url()
                elif class_name == "malware":
                    url = self._generate_malware_url()
                elif class_name == "data_leak":
                    url = self._generate_data_leak_url()
                else:  # scam
                    url = self._generate_scam_url()
                
                urls.append(url)
                labels.append(class_idx)
        
        # Shuffle the dataset
        combined = list(zip(urls, labels))
        random.shuffle(combined)
        urls, labels = zip(*combined)
        urls = list(urls)
        labels = np.array(labels, dtype=np.int32)
        
        # Extract features
        print("Tokenizing URLs...")
        url_tokens = self.tokenizer.tokenize_batch(urls)
        
        print("Extracting features...")
        features = self.extractor.extract_batch(urls)
        
        print(f"Dataset generated: {len(urls)} samples")
        return url_tokens, features, labels, urls
    
    # === URL Generation Methods ===
    
    def _generate_safe_url(self) -> str:
        """Generate a legitimate-looking URL."""
        domain = random.choice(self.SAFE_DOMAINS)
        
        # Randomly add www
        if random.random() < 0.5:
            domain = "www." + domain
        
        # Generate path
        path_options = [
            "",
            "/",
            f"/{self._random_word()}",
            f"/{self._random_word()}/{self._random_word()}",
            f"/search?q={self._random_word()}",
            f"/products/{random.randint(1000, 9999)}",
            f"/article/{self._random_slug()}",
            f"/user/{self._random_word()}",
        ]
        path = random.choice(path_options)
        
        # HTTPS for most safe sites
        scheme = "https" if random.random() < 0.95 else "http"
        
        return f"{scheme}://{domain}{path}"
    
    def _generate_phishing_url(self) -> str:
        """Generate a phishing URL with brand impersonation."""
        brand = random.choice(self.PHISHING_BRANDS)
        
        # Various phishing patterns
        patterns = [
            # Homoglyph substitution
            lambda b: b.replace('o', '0').replace('l', '1').replace('a', '4'),
            # Add numbers
            lambda b: f"{b}{random.randint(1, 99)}",
            # Add keyword
            lambda b: f"{b}-{random.choice(self.PHISHING_KEYWORDS)}",
            # Subdomain attack
            lambda b: f"{b}.{random.choice(self.PHISHING_KEYWORDS)}-secure",
            # Typosquatting
            lambda b: b[:-1] + random.choice('sxz'),
        ]
        
        modified_brand = random.choice(patterns)(brand)
        
        # Use suspicious TLD
        tld = random.choice(list(SUSPICIOUS_TLDS))
        
        # Phishing-style path
        paths = [
            "/login",
            "/signin",
            "/verify",
            "/account/verify",
            "/secure/login",
            f"/auth?id={self._random_hex(8)}",
            f"/{random.choice(self.PHISHING_KEYWORDS)}.php",
            f"/{random.choice(self.PHISHING_KEYWORDS)}.html",
        ]
        
        domain = f"{modified_brand}{tld}"
        path = random.choice(paths)
        
        # Mix of http/https (phishing often lacks HTTPS)
        scheme = "https" if random.random() < 0.4 else "http"
        
        return f"{scheme}://{domain}{path}"
    
    def _generate_malware_url(self) -> str:
        """Generate a malware distribution URL."""
        # Random or suspicious looking domain
        domain_patterns = [
            f"{self._random_string(8, 12)}.{random.choice(list(SUSPICIOUS_TLDS))[1:]}",
            f"download-{self._random_word()}.com",
            f"free-{self._random_word()}.net",
            f"{self._random_word()}-crack.xyz",
            f"{self._random_hex(6)}.top",
        ]
        
        domain = random.choice(domain_patterns)
        
        # Malware-indicative paths
        paths = [
            f"{random.choice(self.MALWARE_PATHS)}{self._random_word()}{random.choice(self.MALWARE_EXTENSIONS)}",
            f"/free/{self._random_word()}_full_version{random.choice(self.MALWARE_EXTENSIONS)}",
            f"/d/{self._random_hex(16)}{random.choice(self.MALWARE_EXTENSIONS)}",
            f"/{self._random_word()}/setup{random.choice(self.MALWARE_EXTENSIONS)}",
        ]
        
        path = random.choice(paths)
        scheme = "http" if random.random() < 0.7 else "https"
        
        return f"{scheme}://{domain}{path}"
    
    def _generate_data_leak_url(self) -> str:
        """Generate a data leak/privacy risk URL."""
        # Generic or suspicious domains
        domain_patterns = [
            f"{self._random_word()}-{self._random_word()}.com",
            f"free{self._random_word()}.net",
            f"{self._random_word()}online.org",
            f"{self._random_string(6, 10)}.info",
        ]
        
        domain = random.choice(domain_patterns)
        
        # Data collection paths
        paths = [
            "/form",
            "/survey",
            "/signup",
            "/register",
            f"/collect?id={self._random_hex(8)}",
            "/submit-data",
            f"/form/{self._random_word()}",
            "/personal-info",
        ]
        
        path = random.choice(paths)
        
        # Often no HTTPS
        scheme = "http" if random.random() < 0.6 else "https"
        
        return f"{scheme}://{domain}{path}"
    
    def _generate_scam_url(self) -> str:
        """Generate a scam/too-good-to-be-true URL."""
        # Scam domain patterns
        domain_patterns = [
            f"{random.choice(self.SCAM_KEYWORDS)}-{self._random_word()}.com",
            f"get-{random.choice(self.SCAM_KEYWORDS)}-now.net",
            f"{self._random_word()}{random.choice(self.SCAM_KEYWORDS)}.xyz",
            f"claim-your-{random.choice(self.SCAM_KEYWORDS)}.top",
        ]
        
        domain = random.choice(domain_patterns)
        
        # Scam paths
        paths = [
            f"/claim?user={self._random_hex(8)}",
            "/winner",
            f"/prize/{random.randint(1000, 9999)}",
            "/congratulations",
            f"/{random.choice(self.SCAM_KEYWORDS)}",
            "/free-gift",
            "/limited-offer",
        ]
        
        path = random.choice(paths)
        scheme = "http" if random.random() < 0.5 else "https"
        
        return f"{scheme}://{domain}{path}"
    
    # === Helper Methods ===
    
    def _random_word(self) -> str:
        """Generate a random word-like string."""
        words = [
            "update", "service", "portal", "online", "support", "help",
            "secure", "user", "account", "center", "info", "data",
            "cloud", "tech", "web", "app", "digital", "smart",
        ]
        return random.choice(words)
    
    def _random_string(self, min_len: int = 5, max_len: int = 10) -> str:
        """Generate a random alphanumeric string."""
        length = random.randint(min_len, max_len)
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _random_hex(self, length: int = 8) -> str:
        """Generate a random hex string."""
        return ''.join(random.choice('0123456789abcdef') for _ in range(length))
    
    def _random_slug(self) -> str:
        """Generate a URL-friendly slug."""
        words = ['best', 'top', 'how', 'what', 'guide', 'tips', 'ways', 'new']
        return '-'.join(random.sample(words, 3))
    
    def save_dataset(
        self, 
        url_tokens: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        urls: List[str],
        prefix: str = "dataset"
    ):
        """Save dataset to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        np.save(DATA_DIR / f"{prefix}_url_tokens.npy", url_tokens)
        np.save(DATA_DIR / f"{prefix}_features.npy", features)
        np.save(DATA_DIR / f"{prefix}_labels.npy", labels)
        
        with open(DATA_DIR / f"{prefix}_urls.txt", 'w') as f:
            f.write('\n'.join(urls))
        
        print(f"Dataset saved to {DATA_DIR}")
    
    def load_dataset(
        self, 
        prefix: str = "dataset"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load dataset from disk."""
        url_tokens = np.load(DATA_DIR / f"{prefix}_url_tokens.npy")
        features = np.load(DATA_DIR / f"{prefix}_features.npy")
        labels = np.load(DATA_DIR / f"{prefix}_labels.npy")
        
        with open(DATA_DIR / f"{prefix}_urls.txt", 'r') as f:
            urls = f.read().strip().split('\n')
        
        return url_tokens, features, labels, urls


if __name__ == "__main__":
    # Generate a small test dataset
    generator = DatasetGenerator(seed=42)
    
    print("Generating test dataset (1000 samples)...")
    url_tokens, features, labels, urls = generator.generate_dataset(n_samples=1000)
    
    print(f"\nDataset shape:")
    print(f"  URL tokens: {url_tokens.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    
    print(f"\nClass distribution:")
    for i, class_name in enumerate(THREAT_CLASSES):
        count = np.sum(labels == i)
        print(f"  {class_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    print("\nSample URLs by class:")
    for i, class_name in enumerate(THREAT_CLASSES):
        idx = np.where(labels == i)[0][0]
        print(f"  {class_name}: {urls[idx]}")
