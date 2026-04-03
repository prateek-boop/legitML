"""
Dual-Branch TensorFlow Neural Network for Threat Detection
Branch 1: Character-level CNN with Attention for URL pattern recognition
Branch 2: Dense network for engineered features
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import MODEL_CONFIG, TRAINING_CONFIG, THREAT_CLASSES, CLASS_WEIGHTS, SAVED_MODEL_DIR
except ImportError:
    MODEL_CONFIG = {
        "url_max_length": 200,
        "char_vocab_size": 74,
        "embedding_dim": 64,  # Increased
        "cnn_filters": 256,   # Increased
        "dnn_units": [512, 256, 128],  # Deeper
        "num_features": 41,
        "num_classes": 5,
        "dropout_rate": 0.4,
    }
    TRAINING_CONFIG = {
        "batch_size": 64,
        "epochs": 50,
        "learning_rate": 0.001,
        "early_stopping_patience": 7,
    }
    THREAT_CLASSES = ["safe", "phishing", "malware", "data_leak", "scam"]
    CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0, 3: 1.8, 4: 2.2}
    SAVED_MODEL_DIR = Path(__file__).parent / "saved_model"


class AttentionLayer(layers.Layer):
    """Self-attention layer for focusing on important URL parts."""
    
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        # x shape: (batch, seq_len, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ThreatDetectionModel:
    """
    Enhanced Dual-Branch Neural Network with Attention for URL Threat Detection.
    
    Architecture v2:
    - Branch 1 (CNN + Attention): Multi-scale character patterns with self-attention
    - Branch 2 (DNN): Deep feature network with residual connections
    - Fusion: Multi-layer fusion with skip connections
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODEL_CONFIG
        self.model: Optional[Model] = None
        self.history = None
        self._build_model()
    
    def _build_model(self):
        """Build the enhanced dual-branch neural network."""
        
        # === Branch 1: Character-level CNN with Attention ===
        url_input = Input(
            shape=(self.config["url_max_length"],), 
            dtype=tf.int32, 
            name="url_input"
        )
        
        # Embedding layer with larger dimensions
        x1 = layers.Embedding(
            input_dim=self.config["char_vocab_size"] + 2,
            output_dim=64,  # Increased embedding dim
            name="char_embedding"
        )(url_input)
        
        # Multi-scale CNN: capture patterns of different lengths
        # Kernel 2: bigrams (http, www, .com)
        conv2 = layers.Conv1D(128, kernel_size=2, activation='relu', padding='same')(x1)
        conv2 = layers.BatchNormalization()(conv2)
        
        # Kernel 3: trigrams (common patterns)
        conv3 = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x1)
        conv3 = layers.BatchNormalization()(conv3)
        
        # Kernel 5: longer patterns (domains, keywords)
        conv5 = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x1)
        conv5 = layers.BatchNormalization()(conv5)
        
        # Kernel 7: even longer patterns
        conv7 = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(x1)
        conv7 = layers.BatchNormalization()(conv7)
        
        # Concatenate multi-scale features
        x1 = layers.Concatenate()([conv2, conv3, conv5, conv7])
        
        # Second conv layer
        x1 = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        
        # Attention mechanism - focus on important parts
        attention_out = AttentionLayer(units=128, name="url_attention")(x1)
        
        # Also keep global max pooling for strongest signals
        max_pool = layers.GlobalMaxPooling1D()(x1)
        avg_pool = layers.GlobalAveragePooling1D()(x1)
        
        # Combine attention + pooling
        x1 = layers.Concatenate()([attention_out, max_pool, avg_pool])
        
        # Dense layers for CNN branch
        x1 = layers.Dense(256, activation='relu')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.4)(x1)
        x1 = layers.Dense(128, activation='relu')(x1)
        
        # === Branch 2: Enhanced DNN for Features ===
        feature_input = Input(
            shape=(self.config["num_features"],), 
            dtype=tf.float32,
            name="feature_input"
        )
        
        # Batch normalization for feature scaling
        x2 = layers.BatchNormalization()(feature_input)
        
        # First dense block
        x2 = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
        
        # Second dense block with residual-like connection
        x2_shortcut = layers.Dense(128)(x2)  # Project for residual
        x2 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
        x2 = layers.Dense(128, activation='relu')(x2)
        x2 = layers.Add()([x2, x2_shortcut])  # Residual connection
        x2 = layers.Activation('relu')(x2)
        
        # Third dense block
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.Dropout(0.4)(x2)
        
        # === Enhanced Fusion Layer ===
        fused = layers.Concatenate(name="fusion")([x1, x2])
        
        # Multi-layer fusion
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer
        output = layers.Dense(
            self.config["num_classes"], 
            activation='softmax',
            name="threat_output"
        )(x)
        
        # Create the model
        self.model = Model(
            inputs=[url_input, feature_input],
            outputs=output,
            name="ShieldNet_v2_Attention"
        )
        
        # Compile with cosine decay
        total_steps = TRAINING_CONFIG.get("epochs", 50) * 1500
        lr_schedule = CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=total_steps,
            alpha=0.01
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def summary(self):
        """Print model architecture summary."""
        if self.model:
            self.model.summary()
    
    def train(
        self,
        url_tokens: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        validation_data: Optional[Tuple] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        class_weights: Optional[Dict] = None,
        save_path: Optional[str] = None,
        resume: bool = True
    ) -> Dict:
        """
        Train the model on provided data.
        
        Args:
            url_tokens: Tokenized URLs, shape (n_samples, url_max_length)
            features: Engineered features, shape (n_samples, num_features)
            labels: Class labels, shape (n_samples,)
            validation_data: Optional tuple of ((url_tokens, features), labels)
            epochs: Training epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            save_path: Path to save the best model
            resume: If True, automatically resume from latest checkpoint
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or TRAINING_CONFIG.get("epochs", 50)
        batch_size = batch_size or TRAINING_CONFIG.get("batch_size", 64)
        class_weights = class_weights or CLASS_WEIGHTS
        save_path = save_path or str(SAVED_MODEL_DIR / "best_model.keras")
        checkpoint_dir = SAVED_MODEL_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing checkpoints and resume
        initial_epoch = 0
        if resume:
            checkpoints = self.list_checkpoints()
            if checkpoints:
                print(f"\n🔄 Found {len(checkpoints)} existing checkpoint(s)")
                latest_epoch = self.load_checkpoint()
                if latest_epoch is not None:
                    initial_epoch = latest_epoch + 1
                    print(f"✅ Resuming from epoch {initial_epoch}")
                    if initial_epoch >= epochs:
                        print(f"⚠️  Already trained {initial_epoch} epochs (requested {epochs})")
                        print(f"   Increase --epochs or delete checkpoints to retrain")
                        return {}
        
        # Callbacks with checkpoint recording
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # Save checkpoint every epoch
            ModelCheckpoint(
                filepath=str(checkpoint_dir / "epoch_{epoch:02d}_acc_{val_accuracy:.4f}.keras"),
                monitor='val_accuracy',
                save_best_only=False,
                save_freq='epoch',
                verbose=0
            ),
            # Early stopping with patience
            EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_CONFIG.get("early_stopping_patience", 10),
                restore_best_weights=True,
                verbose=1
            ),
        ]
        
        print(f"\n📁 Checkpoints will be saved to: {checkpoint_dir}")
        print(f"📁 Best model will be saved to: {save_path}")
        if initial_epoch > 0:
            print(f"🔄 Starting from epoch {initial_epoch + 1}/{epochs}\n")
        else:
            print(f"🆕 Starting fresh training for {epochs} epochs\n")
        
        # Train
        self.history = self.model.fit(
            [url_tokens, features],
            labels,
            validation_data=validation_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, url_tokens: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Predict threat class for URLs.
        
        Returns:
            Class indices (0=safe, 1=phishing, 2=malware, 3=data_leak, 4=scam)
        """
        probabilities = self.model.predict([url_tokens, features], verbose=0)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, url_tokens: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Returns:
            Probability distribution over classes, shape (n_samples, num_classes)
        """
        return self.model.predict([url_tokens, features], verbose=0)
    
    def predict_with_confidence(
        self, 
        url_tokens: np.ndarray, 
        features: np.ndarray
    ) -> List[Dict]:
        """
        Predict with detailed confidence scores.
        
        Returns:
            List of dicts with class, confidence, and all probabilities
        """
        probabilities = self.predict_proba(url_tokens, features)
        results = []
        
        for probs in probabilities:
            class_idx = np.argmax(probs)
            results.append({
                "class": THREAT_CLASSES[class_idx],
                "class_index": int(class_idx),
                "confidence": float(probs[class_idx]),
                "probabilities": {
                    THREAT_CLASSES[i]: float(p) 
                    for i, p in enumerate(probs)
                }
            })
        
        return results
    
    def evaluate(
        self, 
        url_tokens: np.ndarray, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with loss, accuracy, and per-class metrics
        """
        # Basic evaluation
        loss, accuracy = self.model.evaluate(
            [url_tokens, features], 
            labels, 
            verbose=0
        )
        
        # Per-class predictions for detailed metrics
        predictions = self.predict(url_tokens, features)
        
        # Calculate per-class accuracy
        class_metrics = {}
        for i, class_name in enumerate(THREAT_CLASSES):
            mask = labels == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == i)
                class_metrics[class_name] = {
                    "accuracy": float(class_acc),
                    "count": int(np.sum(mask))
                }
        
        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "class_metrics": class_metrics
        }
    
    def save(self, path: Optional[str] = None):
        """Save the model to disk."""
        path = path or str(SAVED_MODEL_DIR / "shieldnet_model.keras")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def load(self, path: Optional[str] = None):
        """Load a saved model from disk."""
        path = path or str(SAVED_MODEL_DIR / "shieldnet_model.keras")
        # Register custom layer for loading
        custom_objects = {'AttentionLayer': AttentionLayer}
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        print(f"Model loaded from: {path}")
    
    @staticmethod
    def list_checkpoints() -> List[str]:
        """List all available checkpoints."""
        checkpoint_dir = SAVED_MODEL_DIR / "checkpoints"
        if not checkpoint_dir.exists():
            return []
        checkpoints = sorted(checkpoint_dir.glob("*.keras"), reverse=True)
        return [str(cp) for cp in checkpoints]
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """
        Load from a specific checkpoint.
        If no path given, loads the latest checkpoint.
        Returns the epoch number from the checkpoint filename.
        """
        if checkpoint_path is None:
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                print("No checkpoints found!")
                return None
            checkpoint_path = checkpoints[0]
            print(f"Loading latest checkpoint: {checkpoint_path}")
        
        self.load(checkpoint_path)
        
        # Extract epoch number from filename (format: epoch_XX_acc_0.XXXX.keras)
        import re
        match = re.search(r'epoch_(\d+)_', checkpoint_path)
        if match:
            return int(match.group(1))
        return None
    
    def get_latest_epoch(self) -> int:
        """Get the epoch number from the latest checkpoint, or 0 if none."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return 0
        import re
        match = re.search(r'epoch_(\d+)_', checkpoints[0])
        if match:
            return int(match.group(1))
        return 0
    
    def get_feature_branch_output(
        self, 
        features: np.ndarray
    ) -> np.ndarray:
        """
        Get intermediate outputs from feature branch (for explainability).
        """
        # Create a sub-model that outputs the feature branch final layer
        feature_input = self.model.get_layer("feature_input").input
        feature_output = self.model.get_layer("dnn_dense3").output
        
        feature_branch = Model(inputs=feature_input, outputs=feature_output)
        return feature_branch.predict(features, verbose=0)


def create_model(config: Optional[Dict] = None) -> ThreatDetectionModel:
    """Factory function to create a new model instance."""
    return ThreatDetectionModel(config)


if __name__ == "__main__":
    # Test model creation
    print("Creating ThreatDetectionModel...")
    model = ThreatDetectionModel()
    model.summary()
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    batch_size = 4
    url_tokens = np.random.randint(0, 74, (batch_size, 200))
    features = np.random.random((batch_size, 41)).astype(np.float32)
    
    predictions = model.predict_with_confidence(url_tokens, features)
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred['class']} (confidence: {pred['confidence']:.2%})")
    
    print("\n✅ Model created successfully!")
