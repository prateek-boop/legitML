"""
Lightweight Quantized Model Detector for ShieldNet
Optimized for fast inference with minimal memory footprint.

Usage:
    from ml_engine.quantized_detector import QuantizedDetector
    
    detector = QuantizedDetector()
    result = detector.predict("https://suspicious-site.com")
    print(result)  # {'class': 'phishing', 'confidence': 0.92, 'all_scores': {...}}
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor

try:
    from config import SAVED_MODEL_DIR, THREAT_CLASSES
except ImportError:
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]


class QuantizedDetector:
    """
    Lightweight threat detector using quantized TFLite model.
    
    Benefits over full Keras model:
    - ~10x smaller model size (0.94 MB vs 10.73 MB)
    - ~3x faster loading time
    - ~50% less memory usage
    - Optimized for CPU inference
    
    Example:
        detector = QuantizedDetector()
        
        # Single prediction
        result = detector.predict("https://example.com")
        print(f"Class: {result['class']}, Confidence: {result['confidence']:.2%}")
        
        # Batch prediction
        results = detector.predict_batch([
            "https://google.com",
            "https://suspicious-login.com/verify"
        ])
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        num_threads: int = 4
    ):
        """
        Initialize the quantized detector.
        
        Args:
            model_path: Path to the .tflite model file. 
                       Defaults to shieldnet_quantized_dynamic.tflite
            num_threads: Number of CPU threads for inference (default: 4)
        """
        self.model_path = model_path or str(
            SAVED_MODEL_DIR / "shieldnet_quantized_dynamic.tflite"
        )
        self.num_threads = num_threads
        self.threat_classes = THREAT_CLASSES
        
        # Initialize components
        self._load_model()
        self.tokenizer = URLTokenizer()
        self.extractor = FeatureExtractor()
        
        # Warm up the model
        self._warmup()
    
    def _load_model(self):
        """Load the TFLite model and allocate tensors."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Quantized model not found: {self.model_path}\n"
                f"Run: python ml_engine/quantize_model.py --type dynamic"
            )
        
        # Create interpreter with optimizations
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path,
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Cache input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Identify which input is which based on shape
        for inp in self.input_details:
            if inp['shape'][-1] == 200:  # URL tokens
                self.url_input_index = inp['index']
            else:  # Features (41)
                self.feature_input_index = inp['index']
    
    def _warmup(self):
        """Warm up the model with a dummy prediction."""
        dummy_url = "https://warmup.example.com"
        self._predict_single(dummy_url)
    
    def _predict_single(self, url: str) -> np.ndarray:
        """Run inference on a single URL, return raw probabilities."""
        # Tokenize and extract features
        url_tokens = self.tokenizer.tokenize(url).reshape(1, -1).astype(np.int32)
        features = self.extractor.extract(url).reshape(1, -1).astype(np.float32)
        
        # Set inputs
        self.interpreter.set_tensor(self.url_input_index, url_tokens)
        self.interpreter.set_tensor(self.feature_input_index, features)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    
    def predict(self, url: str) -> Dict:
        """
        Predict the threat class for a single URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with:
            - class: Predicted threat class (safe, phishing, malware, data_leak, scam)
            - confidence: Confidence score (0-1)
            - all_scores: Dictionary of all class probabilities
            - inference_time_ms: Time taken for inference
        """
        start_time = time.perf_counter()
        probs = self._predict_single(url)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        predicted_idx = np.argmax(probs)
        
        return {
            "url": url,
            "class": self.threat_classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "all_scores": {
                cls: float(prob) 
                for cls, prob in zip(self.threat_classes, probs)
            },
            "inference_time_ms": round(inference_time, 2)
        }
    
    def predict_batch(self, urls: List[str]) -> List[Dict]:
        """
        Predict threat classes for multiple URLs.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(url) for url in urls]
    
    def is_threat(self, url: str, threshold: float = 0.5) -> bool:
        """
        Quick check if a URL is a threat.
        
        Args:
            url: The URL to check
            threshold: Confidence threshold for threat detection
            
        Returns:
            True if the URL is classified as any threat type (not safe)
        """
        result = self.predict(url)
        return result["class"] != "safe" and result["confidence"] >= threshold
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)
        
        return {
            "model_path": self.model_path,
            "model_size_mb": round(model_size, 2),
            "num_threads": self.num_threads,
            "input_shapes": {
                "url_tokens": [200],
                "features": [41]
            },
            "output_classes": self.threat_classes,
            "quantization": "dynamic_range"
        }
    
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark the model's inference speed.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with timing statistics
        """
        test_urls = [
            "https://www.google.com/search?q=test",
            "https://secure-bank-login.suspicious.com/verify",
            "http://192.168.1.1/admin/download.exe",
            "https://amazon.com.fake-deals.net/gift",
            "https://github.com/microsoft/vscode",
        ]
        
        times = []
        for i in range(num_iterations):
            url = test_urls[i % len(test_urls)]
            start = time.perf_counter()
            self._predict_single(url)
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            "num_iterations": num_iterations,
            "avg_ms": round(np.mean(times), 2),
            "min_ms": round(np.min(times), 2),
            "max_ms": round(np.max(times), 2),
            "std_ms": round(np.std(times), 2),
            "throughput_per_sec": round(1000 / np.mean(times), 1)
        }


# Convenience function for quick predictions
def quick_scan(url: str) -> Dict:
    """
    Quick one-liner threat scan.
    Note: Creates a new detector each time - use QuantizedDetector class for batch work.
    
    Example:
        from ml_engine.quantized_detector import quick_scan
        result = quick_scan("https://suspicious-site.com")
    """
    detector = QuantizedDetector()
    return detector.predict(url)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🛡️  ShieldNet Quantized Detector Demo")
    print("=" * 60)
    
    # Initialize detector
    print("\n⏳ Loading quantized model...")
    start = time.perf_counter()
    detector = QuantizedDetector()
    load_time = (time.perf_counter() - start) * 1000
    print(f"✅ Model loaded in {load_time:.0f}ms")
    
    # Show model info
    info = detector.get_model_info()
    print(f"\n📊 Model Info:")
    print(f"   Size: {info['model_size_mb']} MB")
    print(f"   Threads: {info['num_threads']}")
    print(f"   Classes: {', '.join(info['output_classes'])}")
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://secure-login.phishing-site.com/verify",
        "http://malware-download.com/virus.exe",
        "https://github.com/microsoft/vscode",
        "http://free-gift-card-winner.com/claim",
        "https://docs.python.org/3/library/",
    ]
    
    print(f"\n🔍 Scanning {len(test_urls)} URLs...")
    print("-" * 70)
    
    for url in test_urls:
        result = detector.predict(url)
        status = "⚠️ " if result["class"] != "safe" else "✅"
        print(f"{status} {url[:45]:<45} → {result['class']:<10} ({result['confidence']:.1%})")
    
    print("-" * 70)
    
    # Benchmark
    print("\n⚡ Running benchmark (100 iterations)...")
    bench = detector.benchmark(100)
    print(f"   Average: {bench['avg_ms']:.2f}ms per prediction")
    print(f"   Throughput: {bench['throughput_per_sec']:.0f} URLs/second")
    
    print("\n✅ Demo complete!")
