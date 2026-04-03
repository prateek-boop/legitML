"""
ShieldNet Model Training Pipeline
Complete training script with evaluation and visualization

Usage:
    python ml_engine/train_model.py                    # Train on synthetic data
    python ml_engine/train_model.py --dataset real     # Train on real data
    python ml_engine/train_model.py --dataset combined # Train on both
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import our modules
from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor
from ml_engine.model import ThreatDetectionModel
from ml_engine.dataset_generator import DatasetGenerator
from ml_engine.explainer import ThreatExplainer
from ml_engine.real_data_loader import RealDatasetLoader

try:
    from config import (
        THREAT_CLASSES, TRAINING_CONFIG, CLASS_WEIGHTS,
        SAVED_MODEL_DIR, DATA_DIR
    )
except ImportError:
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]
    TRAINING_CONFIG = {
        "batch_size": 64,
        "epochs": 50,
        "learning_rate": 0.001,
        "early_stopping_patience": 5,
        "validation_split": 0.1,
        "test_split": 0.1,
    }
    CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0, 3: 1.8, 4: 2.2}
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"
    DATA_DIR = Path(__file__).parent.parent / "data"


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text: str):
    """Print section header."""
    print(f"\n--- {text} ---")


def generate_or_load_data(
    n_samples: int = 50000,
    force_regenerate: bool = False,
    dataset_type: str = "synthetic"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Generate new dataset or load existing one.
    
    Args:
        n_samples: Number of samples to generate
        force_regenerate: Force regeneration even if cached
        dataset_type: "synthetic", "real", or "combined"
    """
    
    if dataset_type == "real":
        # Use real data from threat intelligence feeds
        return load_real_data(n_samples, force_regenerate)
    elif dataset_type == "combined":
        # Combine real + synthetic
        return load_combined_data(n_samples, force_regenerate)
    else:
        # Default: synthetic data
        return load_synthetic_data(n_samples, force_regenerate)


def load_synthetic_data(n_samples: int, force_regenerate: bool) -> Tuple:
    """Load or generate synthetic dataset."""
    data_files = [
        DATA_DIR / "dataset_url_tokens.npy",
        DATA_DIR / "dataset_features.npy",
        DATA_DIR / "dataset_labels.npy",
        DATA_DIR / "dataset_urls.txt",
    ]
    
    if not force_regenerate and all(f.exists() for f in data_files):
        print_section("Loading cached synthetic dataset")
        generator = DatasetGenerator()
        return generator.load_dataset()
    
    print_section(f"Generating {n_samples:,} synthetic samples")
    generator = DatasetGenerator(seed=42)
    url_tokens, features, labels, urls = generator.generate_dataset(n_samples)
    generator.save_dataset(url_tokens, features, labels, urls)
    
    return url_tokens, features, labels, urls


def load_real_data(n_samples: int, force_regenerate: bool) -> Tuple:
    """Load real data from threat intelligence feeds."""
    data_files = [
        DATA_DIR / "real_dataset_url_tokens.npy",
        DATA_DIR / "real_dataset_features.npy",
        DATA_DIR / "real_dataset_labels.npy",
        DATA_DIR / "real_dataset_urls.txt",
    ]
    
    loader = RealDatasetLoader()
    
    if not force_regenerate and all(f.exists() for f in data_files):
        print_section("Loading cached real dataset")
        return loader.load_dataset(prefix="real_dataset")
    
    print_section(f"Fetching {n_samples:,} real URLs from threat feeds")
    url_tokens, features, labels, urls = loader.build_combined_dataset(max_total=n_samples)
    loader.save_dataset(url_tokens, features, labels, urls, prefix="real_dataset")
    
    return url_tokens, features, labels, urls


def load_combined_data(n_samples: int, force_regenerate: bool) -> Tuple:
    """Combine real and synthetic data for maximum coverage."""
    print_section("Building combined dataset (real + synthetic)")
    
    # Get real data (60% of total)
    real_count = int(n_samples * 0.6)
    print(f"\n📥 Loading {real_count:,} real samples...")
    
    loader = RealDatasetLoader()
    try:
        real_tokens, real_features, real_labels, real_urls = loader.build_combined_dataset(
            max_total=real_count
        )
    except Exception as e:
        print(f"⚠️ Real data fetch failed: {e}")
        print("   Falling back to synthetic data only")
        return load_synthetic_data(n_samples, force_regenerate)
    
    # Get synthetic data (40% of total)
    synthetic_count = n_samples - len(real_labels)
    print(f"\n🔄 Generating {synthetic_count:,} synthetic samples...")
    
    generator = DatasetGenerator(seed=42)
    syn_tokens, syn_features, syn_labels, syn_urls = generator.generate_dataset(synthetic_count)
    
    # Combine
    print("\n🔀 Combining datasets...")
    url_tokens = np.concatenate([real_tokens, syn_tokens], axis=0)
    features = np.concatenate([real_features, syn_features], axis=0)
    labels = np.concatenate([real_labels, syn_labels], axis=0)
    urls = list(real_urls) + list(syn_urls)
    
    # Shuffle
    indices = np.random.permutation(len(labels))
    url_tokens = url_tokens[indices]
    features = features[indices]
    labels = labels[indices]
    urls = [urls[i] for i in indices]
    
    print(f"✅ Combined dataset: {len(labels):,} samples")
    print(f"   Real: {len(real_labels):,} | Synthetic: {synthetic_count:,}")
    
    return url_tokens, features, labels, urls


def split_data(
    url_tokens: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict:
    """Split data into train/val/test sets with stratification."""
    
    print_section("Splitting dataset")
    
    # First split: separate test set
    X_temp_tokens, X_test_tokens, X_temp_feats, X_test_feats, y_temp, y_test = train_test_split(
        url_tokens, features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train_tokens, X_val_tokens, X_train_feats, X_val_feats, y_train, y_val = train_test_split(
        X_temp_tokens, X_temp_feats, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"Training set:   {len(y_train):,} samples")
    print(f"Validation set: {len(y_val):,} samples")
    print(f"Test set:       {len(y_test):,} samples")
    
    return {
        "train": (X_train_tokens, X_train_feats, y_train),
        "val": (X_val_tokens, X_val_feats, y_val),
        "test": (X_test_tokens, X_test_feats, y_test),
    }


def train_model(data: Dict, epochs: int = None) -> Tuple[ThreatDetectionModel, Dict]:
    """Train the model on the provided data."""
    
    print_header("TRAINING MODEL")
    
    X_train_tokens, X_train_feats, y_train = data["train"]
    X_val_tokens, X_val_feats, y_val = data["val"]
    
    # Create model
    print_section("Building model architecture")
    model = ThreatDetectionModel()
    model.summary()
    
    # Train
    print_section("Training")
    epochs = epochs or TRAINING_CONFIG.get("epochs", 50)
    
    history = model.train(
        url_tokens=X_train_tokens,
        features=X_train_feats,
        labels=y_train,
        validation_data=([X_val_tokens, X_val_feats], y_val),
        epochs=epochs,
        batch_size=TRAINING_CONFIG.get("batch_size", 64),
        class_weights=CLASS_WEIGHTS,
    )
    
    return model, history


def evaluate_model(model: ThreatDetectionModel, data: Dict) -> Dict:
    """Comprehensive model evaluation on test set."""
    
    print_header("EVALUATING MODEL")
    
    X_test_tokens, X_test_feats, y_test = data["test"]
    
    # Basic metrics
    print_section("Test Set Performance")
    results = model.evaluate(X_test_tokens, X_test_feats, y_test)
    
    print(f"Loss:     {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Per-class metrics
    print_section("Per-Class Accuracy")
    for class_name, metrics in results['class_metrics'].items():
        print(f"  {class_name:12s}: {metrics['accuracy']:.2%} ({metrics['count']} samples)")
    
    # Detailed classification report - use only classes present in data
    print_section("Classification Report")
    y_pred = model.predict(X_test_tokens, X_test_feats)
    unique_labels = sorted(set(y_test) | set(y_pred))
    present_classes = [THREAT_CLASSES[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=present_classes))
    
    # Confusion matrix
    print_section("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    # Print formatted confusion matrix
    print(f"\n{'':12s}", end='')
    for name in present_classes:
        print(f"{name[:8]:>10s}", end='')
    print()
    
    for i, row in enumerate(cm):
        print(f"{present_classes[i]:12s}", end='')
        for val in row:
            print(f"{val:10d}", end='')
        print()
    
    return results


def test_predictions(model: ThreatDetectionModel):
    """Test model on sample URLs with explanations."""
    
    print_header("TESTING PREDICTIONS")
    
    tokenizer = URLTokenizer()
    extractor = FeatureExtractor()
    explainer = ThreatExplainer()
    
    # Test URLs
    test_urls = [
        "https://www.google.com/search?q=python+tutorial",
        "https://github.com/tensorflow/tensorflow",
        "http://paypa1-secure-verify.xyz/login",
        "http://amaz0n-account-verify.tk/signin",
        "http://free-antivirus-download.top/setup.exe",
        "http://crack-software-free.xyz/download.rar",
        "http://win-iphone-now.xyz/claim?id=abc123",
        "http://congratulations-winner.top/prize",
        "http://192.168.1.1/admin/login.php",
        "http://bit.ly/free-gift-card",
    ]
    
    # Process URLs
    url_tokens = tokenizer.tokenize_batch(test_urls)
    features = extractor.extract_batch(test_urls)
    
    # Get predictions with confidence
    predictions = model.predict_with_confidence(url_tokens, features)
    
    # Generate explanations
    for url, pred, feat in zip(test_urls, predictions, features):
        explanation = explainer.explain(url, pred, feat)
        
        print(f"\n{explanation['icon']} {url}")
        print(f"   → {explanation['threat_level']} ({explanation['category']}) - {explanation['confidence']:.0%}")
        print(f"   Risk Score: {explanation['risk_score']}/100")
        
        if explanation['reasons']:
            print("   Top factors:")
            for reason in explanation['reasons'][:3]:
                print(f"     {reason['icon']} {reason['factor']}")


def save_training_results(model: ThreatDetectionModel, history: Dict, results: Dict):
    """Save model and training artifacts."""
    
    print_header("SAVING RESULTS")
    
    # Save model
    model_path = SAVED_MODEL_DIR / "shieldnet_model.keras"
    model.save(str(model_path))
    
    # Save training history
    import json
    history_path = SAVED_MODEL_DIR / "training_history.json"
    
    # Convert numpy values to Python types for JSON
    json_history = {}
    for key, values in history.items():
        json_history[key] = [float(v) for v in values]
    
    with open(history_path, 'w') as f:
        json.dump(json_history, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    
    # Save final metrics
    metrics_path = SAVED_MODEL_DIR / "final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Metrics saved to: {metrics_path}")


def main(dataset_type: str = "synthetic", n_samples: int = 50000, epochs: int = 30):
    """Main training pipeline."""
    
    print_header("🛡️ SHIELDNET TRAINING PIPELINE")
    print("Deep Learning URL Threat Detection System")
    print(f"Dataset: {dataset_type.upper()}")
    
    # Ensure directories exist
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate or load data
    url_tokens, features, labels, urls = generate_or_load_data(
        n_samples=n_samples,
        force_regenerate=False,
        dataset_type=dataset_type
    )
    
    print(f"\nTotal samples: {len(labels):,}")
    print(f"URL token shape: {url_tokens.shape}")
    print(f"Feature shape: {features.shape}")
    
    # Step 2: Split data
    data = split_data(url_tokens, features, labels)
    
    # Step 3: Train model
    model, history = train_model(data, epochs=epochs)
    
    # Step 4: Evaluate
    results = evaluate_model(model, data)
    
    # Step 5: Test predictions
    test_predictions(model)
    
    # Step 6: Save everything
    save_training_results(model, history, results)
    
    print_header("✅ TRAINING COMPLETE")
    print(f"\nFinal Accuracy: {results['accuracy']:.2%}")
    print(f"Model saved to: {SAVED_MODEL_DIR}")
    
    return model, history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShieldNet threat detection model")
    parser.add_argument(
        "--dataset", 
        choices=["synthetic", "real", "combined"],
        default="synthetic",
        help="Dataset type: synthetic (generated), real (from threat feeds), or combined"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of samples to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs"
    )
    
    args = parser.parse_args()
    
    model, history, results = main(
        dataset_type=args.dataset,
        n_samples=args.samples,
        epochs=args.epochs
    )
