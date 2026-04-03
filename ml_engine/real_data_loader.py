"""
Real Dataset Loader
Fetches real malicious URLs from threat intelligence feeds and combines with safe URLs
"""

import os
import csv
import random
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor

try:
    from config import THREAT_CLASSES, DATA_DIR
except ImportError:
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]
    DATA_DIR = Path(__file__).parent.parent / "data"


class RealDatasetLoader:
    """
    Loads real malicious URLs from multiple threat intelligence sources:
    
    1. PhishTank - Verified phishing URLs (free, requires API key for bulk)
    2. URLhaus - Malware distribution URLs (abuse.ch, free)
    3. OpenPhish - Community phishing feed (free)
    4. Tranco/Alexa Top Sites - Safe URLs baseline
    5. Kaggle Malicious URL Dataset - Pre-labeled dataset
    """
    
    # Data source URLs
    SOURCES = {
        # URLhaus - malware URLs (abuse.ch) - FREE, no API key needed
        "urlhaus_online": "https://urlhaus.abuse.ch/downloads/csv_online/",
        "urlhaus_recent": "https://urlhaus.abuse.ch/downloads/csv_recent/",
        
        # OpenPhish - phishing URLs - FREE
        "openphish": "https://openphish.com/feed.txt",
        
        # PhishTank - verified phishing - FREE (API key for more)
        "phishtank": "http://data.phishtank.com/data/online-valid.csv",
        
        # Tranco Top 1M - safe sites baseline - FREE
        "tranco": "https://tranco-list.eu/top-1m.csv.zip",
        
        # Alternative safe sites from common lists
        "majestic_million": "https://downloads.majestic.com/majestic_million.csv",
    }
    
    # Kaggle datasets (need to download manually or use kaggle API)
    KAGGLE_DATASETS = [
        "sid321axn/malicious-urls-dataset",  # 650k URLs
        "antonyj453/urldataset",              # 450k URLs  
        "xwolf12/malicious-and-benign-websites", 
    ]
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = URLTokenizer()
        self.extractor = FeatureExtractor()
        
        # Cache for downloaded data
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_urlhaus_malware(self, max_urls: int = 10000) -> List[Tuple[str, str]]:
        """
        Fetch malware URLs from URLhaus (abuse.ch).
        Returns list of (url, threat_type) tuples.
        """
        print("📥 Fetching URLhaus malware URLs...")
        urls = []
        
        try:
            # Try online (currently active) URLs first
            response = requests.get(
                self.SOURCES["urlhaus_online"],
                timeout=30,
                headers={"User-Agent": "ShieldNet-ML-Training/1.0"}
            )
            
            if response.status_code == 200:
                lines = response.text.split('\n')
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        url = parts[2].strip('"')
                        threat_type = parts[4].strip('"') if len(parts) > 4 else "malware"
                        if url.startswith('http'):
                            urls.append((url, "malware"))
                    if len(urls) >= max_urls:
                        break
            
            print(f"   ✅ Loaded {len(urls)} malware URLs from URLhaus")
            
        except Exception as e:
            print(f"   ⚠️ URLhaus fetch failed: {e}")
        
        return urls[:max_urls]
    
    def fetch_openphish(self, max_urls: int = 5000) -> List[Tuple[str, str]]:
        """
        Fetch phishing URLs from OpenPhish community feed.
        """
        print("📥 Fetching OpenPhish phishing URLs...")
        urls = []
        
        try:
            response = requests.get(
                self.SOURCES["openphish"],
                timeout=30,
                headers={"User-Agent": "ShieldNet-ML-Training/1.0"}
            )
            
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    url = line.strip()
                    if url.startswith('http'):
                        urls.append((url, "phishing"))
                    if len(urls) >= max_urls:
                        break
            
            print(f"   ✅ Loaded {len(urls)} phishing URLs from OpenPhish")
            
        except Exception as e:
            print(f"   ⚠️ OpenPhish fetch failed: {e}")
        
        return urls[:max_urls]
    
    def fetch_phishtank(self, max_urls: int = 10000) -> List[Tuple[str, str]]:
        """
        Fetch verified phishing URLs from PhishTank.
        Note: May require API key for large downloads.
        """
        print("📥 Fetching PhishTank phishing URLs...")
        urls = []
        
        cache_file = self.cache_dir / "phishtank.csv"
        
        try:
            # Try to download if not cached or cache is old
            if not cache_file.exists():
                response = requests.get(
                    self.SOURCES["phishtank"],
                    timeout=60,
                    headers={"User-Agent": "ShieldNet-ML-Training/1.0"}
                )
                
                if response.status_code == 200:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
            
            # Parse CSV
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        url = row.get('url', '').strip()
                        if url.startswith('http'):
                            urls.append((url, "phishing"))
                        if len(urls) >= max_urls:
                            break
            
            print(f"   ✅ Loaded {len(urls)} phishing URLs from PhishTank")
            
        except Exception as e:
            print(f"   ⚠️ PhishTank fetch failed: {e}")
        
        return urls[:max_urls]
    
    def fetch_safe_urls(self, max_urls: int = 20000) -> List[Tuple[str, str]]:
        """
        Fetch safe/benign URLs from local Majestic Million or Tranco files.
        """
        print("📥 Loading safe URLs from top sites lists...")
        urls = []
        
        # Check for local Majestic Million file first
        majestic_file = self.data_dir / "majestic_million.csv"
        if majestic_file.exists():
            print("   📂 Found majestic_million.csv")
            try:
                with open(majestic_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        domain = row.get('Domain', '').strip()
                        if domain:
                            urls.append((f"https://{domain}", "safe"))
                            urls.append((f"https://www.{domain}", "safe"))
                        if len(urls) >= max_urls:
                            break
                print(f"   ✅ Loaded {len(urls)} safe URLs from Majestic Million")
                return urls[:max_urls]
            except Exception as e:
                print(f"   ⚠️ Error reading Majestic Million: {e}")
        
        # Check for Tranco file
        tranco_file = self.data_dir / "top-1m.csv"
        if tranco_file.exists():
            print("   📂 Found top-1m.csv (Tranco)")
            try:
                with open(tranco_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if ',' in line:
                            parts = line.strip().split(',')
                            domain = parts[-1] if len(parts) > 1 else parts[0]
                            if domain and '.' in domain:
                                urls.append((f"https://{domain}", "safe"))
                        if len(urls) >= max_urls:
                            break
                print(f"   ✅ Loaded {len(urls)} safe URLs from Tranco")
                return urls[:max_urls]
            except Exception as e:
                print(f"   ⚠️ Error reading Tranco: {e}")
        
        # Fallback to hardcoded top domains
        print("   Using hardcoded safe domains...")
        top_safe_domains = [
            "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
            "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
            "microsoft.com", "apple.com", "github.com", "stackoverflow.com", "medium.com",
            "nytimes.com", "bbc.com", "cnn.com", "spotify.com", "dropbox.com",
            "salesforce.com", "adobe.com", "zoom.us", "slack.com", "notion.so",
            "yahoo.com", "bing.com", "duckduckgo.com", "wordpress.com", "blogger.com",
            "tumblr.com", "pinterest.com", "quora.com", "twitch.tv", "discord.com",
            "paypal.com", "stripe.com", "shopify.com", "etsy.com", "ebay.com",
            "walmart.com", "target.com", "bestbuy.com", "homedepot.com", "lowes.com",
            "chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com", "usbank.com",
        ]
        
        paths = ["", "/", "/about", "/contact", "/products", "/services", "/blog", "/help"]
        
        for domain in top_safe_domains:
            for path in paths:
                urls.append((f"https://www.{domain}{path}", "safe"))
                urls.append((f"https://{domain}{path}", "safe"))
                if len(urls) >= max_urls:
                    break
            if len(urls) >= max_urls:
                break
        
        print(f"   ✅ Generated {len(urls)} safe URLs")
        random.shuffle(urls)
        return urls[:max_urls]
    
    def load_kaggle_dataset(self, dataset_path: str = None) -> List[Tuple[str, str]]:
        """
        Load pre-downloaded Kaggle malicious URL dataset.
        
        Supports:
        - malicious_phish.csv (from sid321axn/malicious-urls-dataset)
        - data.csv (from antonyj453/urldataset)
        - Files in archive/ subfolders
        
        Download from Kaggle and place in data/ folder.
        """
        print("📥 Looking for Kaggle datasets...")
        urls = []
        
        # Search patterns - including subfolders from extracted zips
        search_paths = [
            self.data_dir / "malicious_phish.csv",
            self.data_dir / "data.csv",
            self.data_dir / "urldata.csv",
            self.data_dir / "archive" / "malicious_phish.csv",
            self.data_dir / "archive (1)" / "data.csv",
            self.data_dir / "archive" / "data.csv",
        ]
        
        # Also search any subdirectory
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                for csv_file in subdir.glob("*.csv"):
                    if csv_file not in search_paths:
                        search_paths.append(csv_file)
        
        found_files = []
        for filepath in search_paths:
            if filepath.exists() and filepath.name not in ['majestic_million.csv', 'top-1m.csv']:
                found_files.append(filepath)
        
        for filepath in found_files:
            print(f"   📂 Found {filepath.name}")
            urls.extend(self._parse_kaggle_csv(filepath))
        
        if not urls:
            print("   ℹ️ No Kaggle datasets found.")
            print("   💡 Download from: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset")
            print(f"   📁 Place CSV files in: {self.data_dir}")
        else:
            print(f"   ✅ Loaded {len(urls)} URLs from Kaggle datasets")
        
        return urls
    
    def _parse_kaggle_csv(self, filepath: Path) -> List[Tuple[str, str]]:
        """Parse common Kaggle CSV formats."""
        urls = []
        
        # Label mappings from various datasets
        label_map = {
            # sid321axn/malicious-urls-dataset format (malicious_phish.csv)
            "benign": "safe",
            "phishing": "phishing", 
            "malware": "malware",
            "defacement": "malware",  # Treat defacement as malware
            
            # antonyj453/urldataset format (data.csv)
            "good": "safe",
            "bad": "phishing",  # Generic bad = phishing
            
            # Generic labels
            "legitimate": "safe",
            "malicious": "phishing",
            "safe": "safe",
            "0": "safe",
            "1": "phishing",
            "spam": "scam",
        }
        
        try:
            # Count lines for progress
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f) - 1
            print(f"      Processing {total_lines:,} rows...")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    # Find URL column
                    url = None
                    for col in ['url', 'URL', 'urls', 'domain', 'Domain', 'uri', 'URI']:
                        if col in row:
                            url = row[col].strip()
                            break
                    
                    # Find label column
                    label = None
                    for col in ['type', 'Type', 'label', 'Label', 'class', 'Class', 'result', 'Result', 'category']:
                        if col in row:
                            raw_label = row[col].strip().lower()
                            label = label_map.get(raw_label, None)
                            if label is None and raw_label:
                                # Try to infer from partial match
                                if 'phish' in raw_label:
                                    label = 'phishing'
                                elif 'malware' in raw_label or 'virus' in raw_label:
                                    label = 'malware'
                                elif 'safe' in raw_label or 'benign' in raw_label or 'legit' in raw_label:
                                    label = 'safe'
                                elif 'scam' in raw_label or 'spam' in raw_label:
                                    label = 'scam'
                                else:
                                    label = 'phishing'  # Default unknown malicious to phishing
                            break
                    
                    if url and label:
                        # Clean URL
                        url = url.strip()
                        if not url.startswith('http'):
                            url = 'http://' + url
                        urls.append((url, label))
                    
                    # Progress indicator
                    if (i + 1) % 100000 == 0:
                        print(f"      Processed {i+1:,} rows...")
                        
        except Exception as e:
            print(f"   ⚠️ Error parsing {filepath}: {e}")
        
        return urls
    
    def build_combined_dataset(
        self,
        max_total: int = 100000,
        distribution: Dict[str, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Build a combined dataset from all real sources.
        
        Args:
            max_total: Maximum total samples
            distribution: Target class distribution (will resample to match)
        
        Returns:
            url_tokens, features, labels, urls
        """
        print("\n" + "=" * 60)
        print(" 🌐 BUILDING REAL-WORLD DATASET")
        print("=" * 60)
        
        # Default distribution
        distribution = distribution or {
            "safe": 0.40,
            "phishing": 0.35,
            "malware": 0.20,
            "data_leak": 0.03,
            "scam": 0.02,
        }
        
        # Fetch from all sources
        all_urls = {
            "safe": [],
            "phishing": [],
            "malware": [],
            "data_leak": [],
            "scam": [],
        }
        
        # 1. Fetch safe URLs
        safe_urls = self.fetch_safe_urls(max_urls=int(max_total * 0.5))
        all_urls["safe"].extend([u[0] for u in safe_urls])
        
        # 2. Fetch malware URLs from URLhaus
        malware_urls = self.fetch_urlhaus_malware(max_urls=int(max_total * 0.3))
        all_urls["malware"].extend([u[0] for u in malware_urls])
        
        # 3. Fetch phishing URLs
        openphish_urls = self.fetch_openphish(max_urls=int(max_total * 0.2))
        all_urls["phishing"].extend([u[0] for u in openphish_urls])
        
        phishtank_urls = self.fetch_phishtank(max_urls=int(max_total * 0.2))
        all_urls["phishing"].extend([u[0] for u in phishtank_urls])
        
        # 4. Load Kaggle datasets if available
        kaggle_urls = self.load_kaggle_dataset()
        for url, label in kaggle_urls:
            if label in all_urls:
                all_urls[label].append(url)
        
        # Remove duplicates
        for category in all_urls:
            all_urls[category] = list(set(all_urls[category]))
        
        # Print stats
        print("\n📊 Raw data collected:")
        total_raw = 0
        for category, urls in all_urls.items():
            print(f"   {category}: {len(urls):,} URLs")
            total_raw += len(urls)
        print(f"   Total: {total_raw:,} URLs")
        
        # Balance dataset according to distribution
        print("\n⚖️ Balancing dataset...")
        balanced_urls = []
        balanced_labels = []
        
        for category, proportion in distribution.items():
            target_count = int(max_total * proportion)
            available = all_urls.get(category, [])
            
            if len(available) >= target_count:
                # Randomly sample
                sampled = random.sample(available, target_count)
            else:
                # Use all available + oversample if needed
                sampled = available.copy()
                if len(sampled) > 0 and len(sampled) < target_count:
                    # Oversample with replacement
                    sampled.extend(random.choices(available, k=target_count - len(sampled)))
            
            class_idx = THREAT_CLASSES.index(category)
            balanced_urls.extend(sampled)
            balanced_labels.extend([class_idx] * len(sampled))
        
        # Shuffle
        combined = list(zip(balanced_urls, balanced_labels))
        random.shuffle(combined)
        balanced_urls, balanced_labels = zip(*combined)
        balanced_urls = list(balanced_urls)
        balanced_labels = np.array(balanced_labels, dtype=np.int32)
        
        # Extract features
        print("\n🔄 Processing URLs...")
        print("   Tokenizing...")
        url_tokens = self.tokenizer.tokenize_batch(balanced_urls)
        
        print("   Extracting features...")
        features = self.extractor.extract_batch(balanced_urls)
        
        # Final stats
        print("\n✅ Final dataset:")
        print(f"   Total samples: {len(balanced_urls):,}")
        print(f"   URL tokens shape: {url_tokens.shape}")
        print(f"   Features shape: {features.shape}")
        
        print("\n📊 Class distribution:")
        for i, class_name in enumerate(THREAT_CLASSES):
            count = np.sum(balanced_labels == i)
            pct = count / len(balanced_labels) * 100
            print(f"   {class_name}: {count:,} ({pct:.1f}%)")
        
        return url_tokens, features, balanced_labels, balanced_urls
    
    def save_dataset(
        self,
        url_tokens: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        urls: List[str],
        prefix: str = "real_dataset"
    ):
        """Save the dataset to disk."""
        print(f"\n💾 Saving dataset to {self.data_dir}...")
        
        np.save(self.data_dir / f"{prefix}_url_tokens.npy", url_tokens)
        np.save(self.data_dir / f"{prefix}_features.npy", features)
        np.save(self.data_dir / f"{prefix}_labels.npy", labels)
        
        with open(self.data_dir / f"{prefix}_urls.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(urls))
        
        print(f"   ✅ Saved to {self.data_dir}/{prefix}_*")
    
    def load_dataset(
        self,
        prefix: str = "real_dataset"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load dataset from disk."""
        url_tokens = np.load(self.data_dir / f"{prefix}_url_tokens.npy")
        features = np.load(self.data_dir / f"{prefix}_features.npy")
        labels = np.load(self.data_dir / f"{prefix}_labels.npy")
        
        with open(self.data_dir / f"{prefix}_urls.txt", 'r', encoding='utf-8') as f:
            urls = f.read().strip().split('\n')
        
        return url_tokens, features, labels, urls


def download_kaggle_instructions():
    """Print instructions for downloading Kaggle datasets."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          📥 DOWNLOAD REAL DATASETS FROM KAGGLE                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  For best accuracy, download these datasets and place in data/   ║
║                                                                  ║
║  1. BEST: Malicious URLs Dataset (650K+ URLs)                    ║
║     https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
║     → Download 'malicious_phish.csv'                             ║
║                                                                  ║
║  2. URL Dataset (450K URLs)                                      ║
║     https://www.kaggle.com/datasets/antonyj453/urldataset        ║
║     → Download 'urldata.csv'                                     ║
║                                                                  ║
║  3. PhishStorm Dataset                                           ║
║     https://www.kaggle.com/datasets/naufalsaifullah/phishstorm-urls
║                                                                  ║
║  Place downloaded CSV files in:                                  ║
║  C:\\Users\\prate\\OneDrive\\Desktop\\legit0\\data\\              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("🛡️ ShieldNet Real Dataset Loader")
    print()
    
    # Show download instructions
    download_kaggle_instructions()
    
    # Build dataset from online sources
    loader = RealDatasetLoader()
    url_tokens, features, labels, urls = loader.build_combined_dataset(max_total=50000)
    
    # Save
    loader.save_dataset(url_tokens, features, labels, urls)
    
    print("\n✅ Real dataset ready for training!")
    print("   Run: python ml_engine/train_model.py --dataset real")
