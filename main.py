"""
ShieldNet - ML-Powered Malicious Website Detection System

Usage:
    # Train the model (synthetic data - fast for testing)
    python main.py train
    
    # Train with REAL data (recommended for production)
    python main.py train --dataset real
    
    # Train with combined real + synthetic data (best results)
    python main.py train --dataset combined
    
    # Resume training from last checkpoint (automatic by default)
    python main.py train --epochs 50
    
    # Start fresh (ignore checkpoints)
    python main.py train --fresh --epochs 50
    
    # Start the API server
    python main.py serve
    
    # Test a URL
    python main.py test "http://suspicious-site.xyz/login"
    
    # Download real datasets info
    python main.py download-data
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def train(dataset_type: str = "synthetic", samples: int = 50000, epochs: int = 30, fresh: bool = False):
    """Train the ShieldNet model."""
    print("🛡️ Starting ShieldNet Training...")
    print(f"   Dataset: {dataset_type}")
    print(f"   Samples: {samples:,}")
    print(f"   Epochs: {epochs}")
    print(f"   Fresh: {fresh}")
    print()
    
    from ml_engine.train_model import main as train_main
    train_main(dataset_type=dataset_type, n_samples=samples, epochs=epochs, fresh=fresh)


def download_data():
    """Show instructions for downloading real datasets."""
    from ml_engine.real_data_loader import download_kaggle_instructions, RealDatasetLoader
    
    download_kaggle_instructions()
    
    print("\nAttempting to fetch online data sources...")
    loader = RealDatasetLoader()
    
    # Try fetching from free online sources
    loader.fetch_openphish(max_urls=100)
    loader.fetch_urlhaus_malware(max_urls=100)
    loader.fetch_safe_urls(max_urls=100)
    
    print("\n✅ Online sources are accessible!")
    print("\nTo build a full real dataset, run:")
    print("   python main.py train --dataset real")


def serve():
    """Start the API server."""
    print("🚀 Starting ShieldNet API Server...")
    import uvicorn
    from config import API_CONFIG
    
    uvicorn.run(
        "api.server:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=True
    )


def test_url(url: str):
    """Test a single URL."""
    print(f"🔍 Analyzing: {url}")
    print()
    
    from ml_engine.url_tokenizer import URLTokenizer
    from ml_engine.feature_extractor import FeatureExtractor
    from ml_engine.model import ThreatDetectionModel
    from ml_engine.explainer import ThreatExplainer
    from config import SAVED_MODEL_DIR
    
    # Load components
    tokenizer = URLTokenizer()
    extractor = FeatureExtractor()
    explainer = ThreatExplainer()
    model = ThreatDetectionModel()
    
    model_path = SAVED_MODEL_DIR / "shieldnet_model.keras"
    if model_path.exists():
        model.load(str(model_path))
    else:
        print("⚠️ No trained model found. Using untrained model.")
        print("   Run 'python main.py train' first.")
    
    # Analyze
    url_tokens = tokenizer.tokenize_batch([url])
    features = extractor.extract_batch([url])
    
    prediction = model.predict_with_confidence(url_tokens, features)[0]
    explanation = explainer.explain(url, prediction, features[0])
    
    # Display results
    print(f"{explanation['icon']} Threat Level: {explanation['threat_level']}")
    print(f"   Category: {explanation['category']}")
    print(f"   Confidence: {explanation['confidence']:.0%}")
    print(f"   Risk Score: {explanation['risk_score']}/100")
    print(f"   Blocked: {explanation['blocked']}")
    print()
    print("Risk Factors:")
    for reason in explanation['reasons']:
        print(f"   {reason['icon']} {reason['factor']}")
    print()
    print(f"Recommendation: {explanation['recommendation']}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        # Parse training options
        dataset_type = "synthetic"
        samples = 50000
        epochs = 30
        fresh = False
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--dataset" and i + 1 < len(sys.argv):
                dataset_type = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--samples" and i + 1 < len(sys.argv):
                samples = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--epochs" and i + 1 < len(sys.argv):
                epochs = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--fresh":
                fresh = True
                i += 1
            else:
                i += 1
        
        train(dataset_type=dataset_type, samples=samples, epochs=epochs, fresh=fresh)
        
    elif command == "serve":
        serve()
        
    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python main.py test <url>")
            return
        test_url(sys.argv[2])
        
    elif command == "download-data":
        download_data()
        
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
