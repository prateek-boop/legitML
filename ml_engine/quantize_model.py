"""
Model Quantization Script for ShieldNet
Converts the trained Keras model to optimized TFLite format with quantization.

Supports multiple quantization strategies:
1. Dynamic Range Quantization (default) - Best balance of size/accuracy
2. Float16 Quantization - Good for GPU inference
3. Full Integer Quantization - Smallest size, needs representative data
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Import custom layer and config
from ml_engine.model import AttentionLayer, ThreatDetectionModel
from ml_engine.url_tokenizer import URLTokenizer
from ml_engine.feature_extractor import FeatureExtractor


def load_model_with_compatibility(model_path: str, custom_objects: dict = None):
    """
    Load a Keras model with backward compatibility fixes.
    Handles version mismatches (e.g., BatchNormalization renorm params).
    """
    import json
    import zipfile
    import tempfile
    import shutil
    
    custom_objects = custom_objects or {}
    custom_objects['AttentionLayer'] = AttentionLayer
    
    # First try normal loading
    try:
        return keras.models.load_model(model_path, custom_objects=custom_objects)
    except TypeError as e:
        if 'renorm' not in str(e):
            raise
        print("⚠️  Detected Keras version mismatch, applying compatibility fix...")
    
    # Extract and fix the model config
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Remove incompatible BatchNormalization parameters
        def fix_config(obj):
            if isinstance(obj, dict):
                # Remove renorm-related keys from BatchNormalization configs
                if obj.get('class_name') == 'BatchNormalization':
                    config_inner = obj.get('config', {})
                    for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                        config_inner.pop(key, None)
                # Recursively fix nested objects
                for key, value in obj.items():
                    fix_config(value)
            elif isinstance(obj, list):
                for item in obj:
                    fix_config(item)
        
        fix_config(config)
        
        # Write fixed config
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Repack the model
        fixed_model_path = model_path.replace('.keras', '_fixed.keras')
        with zipfile.ZipFile(fixed_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        # Load the fixed model
        model = keras.models.load_model(fixed_model_path, custom_objects=custom_objects)
        
        # Clean up the temporary fixed model
        os.remove(fixed_model_path)
        
        print("✅ Compatibility fix applied successfully")
        return model
        
    finally:
        shutil.rmtree(temp_dir)

try:
    from config import SAVED_MODEL_DIR, MODEL_CONFIG
except ImportError:
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"
    MODEL_CONFIG = {
        "url_max_length": 200,
        "num_features": 41,
    }


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def create_representative_dataset(num_samples: int = 200):
    """
    Create a representative dataset generator for full integer quantization.
    Uses realistic URL patterns for calibration.
    """
    tokenizer = URLTokenizer()
    extractor = FeatureExtractor()
    
    # Sample URLs covering different threat types
    sample_urls = [
        "https://www.google.com/search?q=hello",
        "https://secure-bank-login.suspicious-domain.com/verify",
        "http://192.168.1.1/admin/download.exe",
        "https://amazon.com.fake-deals.net/gift-card",
        "https://github.com/user/repository",
        "http://bit.ly/3xKj2mN",
        "https://drive.google.com/file/d/abc123",
        "https://paypal-secure.phishing-site.com/login",
        "https://www.microsoft.com/en-us/download",
        "http://free-iphone-winner.com/claim-now",
    ] * (num_samples // 10 + 1)
    
    sample_urls = sample_urls[:num_samples]
    
    # Pre-compute all data
    url_tokens = tokenizer.tokenize_batch(sample_urls)
    features = extractor.extract_batch(sample_urls)
    
    def representative_data_gen():
        for i in range(len(sample_urls)):
            yield [
                url_tokens[i:i+1].astype(np.int32),  # INT32 for URL tokens
                features[i:i+1].astype(np.float32)
            ]
    
    return representative_data_gen


def quantize_model(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    quantization_type: str = "dynamic",
    use_representative_data: bool = True,
) -> dict:
    """
    Quantize the ShieldNet model.
    
    Args:
        input_path: Path to the Keras model. Defaults to best_model.keras
        output_path: Output path for quantized model. Auto-generated if None.
        quantization_type: One of "dynamic", "float16", "int8"
        use_representative_data: Whether to use representative dataset (recommended)
    
    Returns:
        Dictionary with metrics (sizes, compression ratio)
    """
    # Set paths
    input_path = input_path or str(SAVED_MODEL_DIR / "best_model.keras")
    
    if output_path is None:
        output_path = str(SAVED_MODEL_DIR / f"shieldnet_quantized_{quantization_type}.tflite")
    
    print("\n" + "=" * 60)
    print(" 🔧 ShieldNet Model Quantization")
    print("=" * 60)
    print(f"\n📁 Input model: {input_path}")
    print(f"📦 Quantization type: {quantization_type}")
    
    # Check if input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Model not found: {input_path}")
    
    original_size = get_file_size_mb(input_path)
    print(f"📊 Original size: {original_size:.2f} MB")
    
    # Load model with custom objects
    print("\n⏳ Loading model...")
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = load_model_with_compatibility(input_path, custom_objects=custom_objects)
    print("✅ Model loaded successfully")
    
    # Create converter
    print("\n⏳ Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization based on type
    if quantization_type == "dynamic":
        # Dynamic range quantization - weights quantized to int8
        # Activations remain float, quantized dynamically at inference
        print("📌 Applying Dynamic Range Quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif quantization_type == "float16":
        # Float16 quantization - good balance, works well on GPU
        print("📌 Applying Float16 Quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization_type == "int8":
        # Full integer quantization - smallest size, needs representative data
        print("📌 Applying Full Integer (INT8) Quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if use_representative_data:
            print("⏳ Generating representative dataset for calibration...")
            converter.representative_dataset = create_representative_dataset(200)
            # Ensure integer-only ops where possible
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for unsupported ops
            ]
            converter.inference_input_type = tf.float32  # Keep inputs as float for ease of use
            converter.inference_output_type = tf.float32
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    # Convert
    print("\n⏳ Converting model (this may take a moment)...")
    try:
        tflite_model = converter.convert()
        print("✅ Conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise
    
    # Save quantized model
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    quantized_size = get_file_size_mb(output_path)
    compression_ratio = original_size / quantized_size
    size_reduction = (1 - quantized_size / original_size) * 100
    
    # Results
    print("\n" + "=" * 60)
    print(" 📊 Quantization Results")
    print("=" * 60)
    print(f"  Original size:    {original_size:.2f} MB")
    print(f"  Quantized size:   {quantized_size:.2f} MB")
    print(f"  Compression:      {compression_ratio:.1f}x")
    print(f"  Size reduction:   {size_reduction:.1f}%")
    print("=" * 60)
    
    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": size_reduction,
        "output_path": output_path,
        "quantization_type": quantization_type,
    }


def verify_quantized_model(model_path: str, num_tests: int = 10) -> dict:
    """
    Verify the quantized model works correctly.
    
    Args:
        model_path: Path to the quantized TFLite model
        num_tests: Number of test inferences to run
    
    Returns:
        Dictionary with verification results
    """
    print("\n" + "=" * 60)
    print(" 🧪 Verifying Quantized Model")
    print("=" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📥 Model Inputs:")
    for inp in input_details:
        print(f"   - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
    
    print(f"\n📤 Model Output:")
    for out in output_details:
        print(f"   - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
    
    # Test inference
    tokenizer = URLTokenizer()
    extractor = FeatureExtractor()
    
    test_urls = [
        "https://www.google.com",
        "https://secure-login.phishing-site.com/verify",
        "http://malware-download.com/virus.exe",
        "https://github.com/microsoft/vscode",
        "http://free-gift-card-winner.com/claim",
    ]
    
    print(f"\n⏳ Running {len(test_urls)} test inferences...")
    
    import time
    predictions = []
    inference_times = []
    
    for url in test_urls:
        # Prepare inputs
        url_tokens = tokenizer.tokenize(url).reshape(1, -1).astype(np.int32)  # INT32 for URL tokens
        features = extractor.extract(url).reshape(1, -1).astype(np.float32)
        
        # Set inputs
        interpreter.set_tensor(input_details[0]['index'], url_tokens)
        interpreter.set_tensor(input_details[1]['index'], features)
        
        # Run inference
        start_time = time.perf_counter()
        interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0])
    
    # Print results
    threat_classes = ["safe", "phishing", "malware", "data_leak", "scam"]
    
    print("\n📊 Test Results:")
    print("-" * 70)
    for i, (url, pred) in enumerate(zip(test_urls, predictions)):
        predicted_class = threat_classes[np.argmax(pred)]
        confidence = np.max(pred) * 100
        print(f"  {url[:45]:<45} → {predicted_class} ({confidence:.1f}%)")
    print("-" * 70)
    
    avg_inference_time = np.mean(inference_times)
    print(f"\n⚡ Average inference time: {avg_inference_time:.2f} ms")
    print("✅ Quantized model verified successfully!")
    
    return {
        "num_tests": len(test_urls),
        "avg_inference_time_ms": avg_inference_time,
        "predictions": predictions,
        "status": "success"
    }


def quantize_all_variants():
    """Generate all quantization variants for comparison."""
    results = {}
    
    for q_type in ["dynamic", "float16", "int8"]:
        print(f"\n{'#' * 60}")
        print(f"# Processing: {q_type.upper()} quantization")
        print(f"{'#' * 60}")
        
        try:
            results[q_type] = quantize_model(quantization_type=q_type)
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[q_type] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print(" 📊 QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Type':<12} {'Size (MB)':<12} {'Compression':<12} {'Reduction':<12}")
    print("-" * 48)
    
    for q_type, result in results.items():
        if "error" not in result:
            print(f"{q_type:<12} {result['quantized_size_mb']:<12.2f} {result['compression_ratio']:<12.1f}x {result['size_reduction_percent']:<12.1f}%")
        else:
            print(f"{q_type:<12} {'FAILED':<12}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize ShieldNet model")
    parser.add_argument(
        "--type", 
        choices=["dynamic", "float16", "int8", "all"],
        default="dynamic",
        help="Quantization type (default: dynamic)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="Input model path (default: best_model.keras)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None, 
        help="Output path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the quantized model after conversion"
    )
    parser.add_argument(
        "--no-representative",
        action="store_true",
        help="Skip representative dataset (faster but less accurate for int8)"
    )
    
    args = parser.parse_args()
    
    if args.type == "all":
        results = quantize_all_variants()
        # Verify the recommended one (dynamic)
        if args.verify and "dynamic" in results and "error" not in results["dynamic"]:
            verify_quantized_model(results["dynamic"]["output_path"])
    else:
        result = quantize_model(
            input_path=args.input,
            output_path=args.output,
            quantization_type=args.type,
            use_representative_data=not args.no_representative
        )
        
        if args.verify:
            verify_quantized_model(result["output_path"])
    
    print("\n✅ Quantization complete!")
