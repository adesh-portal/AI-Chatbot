"""
Enhanced Chatbot Model Training Script
Handles data preprocessing, model training, and saving trained components with advanced logging and neural network improvements.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import re
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# CPU optimization settings
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available CPU cores
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available CPU cores
tf.config.run_functions_eagerly(False)  # Use graph mode for better CPU performance

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedChatbotTrainer:
    def __init__(self, intents_file='intents.json', model_dir='models'):
        self.intents_file = intents_file
        self.model_dir = model_dir
        self.max_sequence_length = 50
        self.vocab_size = 10000
        
        # Training metrics
        self.training_start_time = None
        self.training_history = None
        self.best_accuracy = 0.0
        self.training_stats = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        
        logger.info(f"Enhanced Chatbot Trainer initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Max sequence length: {self.max_sequence_length}")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
    def load_intents(self):
        """Load intents from JSON file with enhanced error handling"""
        logger.info(f"Loading intents from {self.intents_file}")
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            data = None
            
            for encoding in encodings:
                try:
                    with open(self.intents_file, 'r', encoding=encoding) as f:
                        data = json.load(f)
                        logger.info(f"Successfully loaded intents with {encoding} encoding")
                        break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to load with {encoding} encoding, trying next...")
                    continue
            
            if data is None:
                raise ValueError("Could not load intents file with any supported encoding")
            
            if isinstance(data, dict):
                intents = data.get('intents', data.get('intent', []))
            else:
                intents = data
            
            logger.info(f"Loaded {len(intents)} intents")
            
            # Log intent statistics
            total_patterns = sum(len(intent.get('patterns', [])) for intent in intents)
            total_responses = sum(len(intent.get('responses', [])) for intent in intents)
            
            logger.info(f"Total patterns: {total_patterns}")
            logger.info(f"Total responses: {total_responses}")
            logger.info(f"Average patterns per intent: {total_patterns / len(intents):.2f}")
            logger.info(f"Average responses per intent: {total_responses / len(intents):.2f}")
            
            return intents
            
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            raise
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing with better cleaning"""
        if not text:
            return ""
        
        logger.debug(f"Preprocessing text: {text[:50]}...")
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Expand contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is",
            "i'm": "i am", "you're": "you are", "we're": "we are",
            "they're": "they are", "he's": "he is", "she's": "she is",
            "it's": "it is", "that's": "that is", "there's": "there is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\?\!\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self):
        """Prepare training data from intents with enhanced validation"""
        logger.info("Preparing training data...")
        intents = self.load_intents()
        
        patterns = []
        labels = []
        intent_stats = {}
        
        for intent in intents:
            tag = intent['tag']
            patterns_count = 0
            
            for pattern in intent['patterns']:
                # Preprocess each pattern
                cleaned_pattern = self.preprocess_text(pattern)
                if cleaned_pattern and len(cleaned_pattern.split()) > 0:  # Only add non-empty patterns
                    patterns.append(cleaned_pattern)
                    labels.append(tag)
                    patterns_count += 1
            
            intent_stats[tag] = patterns_count
            logger.debug(f"Intent '{tag}': {patterns_count} patterns")
        
        logger.info(f"Loaded {len(patterns)} patterns across {len(set(labels))} intents")
        
        # Log intent distribution
        for tag, count in sorted(intent_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {tag}: {count} patterns")
        
        # Check for class imbalance
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 5:
            logger.warning("High class imbalance detected! Consider data augmentation or class weighting.")
        
        return patterns, labels
    
    def create_tokenizer(self, patterns):
        """Create and fit tokenizer on patterns"""
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        self.tokenizer.fit_on_texts(patterns)
        logger.info(f"Tokenizer created with vocabulary size: {len(self.tokenizer.word_index)}")
        
        return self.tokenizer
    
    def encode_labels(self, labels):
        """Encode string labels to integers"""
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Label encoder created for {len(self.label_encoder.classes_)} classes")
        return encoded_labels
    
    def create_sequences(self, patterns):
        """Convert text patterns to sequences"""
        sequences = self.tokenizer.texts_to_sequences(patterns)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length, 
            padding='post'
        )
        
        logger.info(f"Created sequences with shape: {padded_sequences.shape}")
        return padded_sequences
    
    def build_model(self, num_classes):
        """Build CPU-optimized neural network model"""
        logger.info(f"Building CPU-optimized model for {num_classes} classes")
        
        model = Sequential()
        
        # CPU-optimized embedding layer (reduced dimension)
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=128,  # Reduced from 256 for CPU efficiency
            input_length=self.max_sequence_length,
            mask_zero=True,
            name='embedding'
        ))
        
        # Single LSTM layer (reduced complexity for CPU)
        model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, name='lstm_1'))
        model.add(BatchNormalization(name='batch_norm_1'))
        
        # Dense layers (reduced size for CPU efficiency)
        model.add(Dense(64, activation='relu', name='dense_1'))
        model.add(BatchNormalization(name='batch_norm_2'))
        model.add(Dropout(0.4, name='dropout_1'))
        
        model.add(Dense(32, activation='relu', name='dense_2'))
        model.add(Dropout(0.3, name='dropout_2'))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax', name='output'))
        
        # CPU-optimized optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.002,  # Slightly higher for faster convergence
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Build the model explicitly before compiling
        model.build(input_shape=(None, self.max_sequence_length))
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("CPU-optimized model architecture created")
        logger.info(f"Total parameters: {model.count_params():,}")
        model.summary()
        
        return model
    
    def train_model(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """CPU-optimized model training"""
        logger.info("Starting CPU-optimized model training...")
        self.training_start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Validation set size: {X_val.shape[0]}")
        logger.info(f"Training epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # CPU-optimized callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Less aggressive reduction
                patience=8,  # Reduced patience
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            CSVLogger(
                filename=os.path.join(self.model_dir, 'training_log.csv'),
                append=False
            )
        ]
        
        # Train model
        # Compute class weights to handle imbalance
        logger.info("Computing class weights for imbalanced classes...")
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {int(cls): float(w) for cls, w in zip(classes, class_weights_array)}
        logger.info(f"Class weights computed for {len(class_weight)} classes")

        # Train model
        logger.info("Training started...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
        )
        
        # Store training history
        self.training_history = history
        
        # Evaluate model
        logger.info("Evaluating model...")
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Update best accuracy
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
        
        # Log results
        training_time = time.time() - self.training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")
        
        # Store training stats
        self.training_stats = {
            'final_accuracy': val_accuracy,
            'best_accuracy': self.best_accuracy,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'final_loss': val_loss
        }
        
        return history
    
    def plot_training_history(self):
        """Plot training history for visualization"""
        if self.training_history is None:
            logger.warning("No training history available for plotting")
            return
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training History', fontsize=16)
            
            # Plot accuracy
            axes[0, 0].plot(self.training_history.history['accuracy'], label='Training Accuracy')
            axes[0, 0].plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot loss
            axes[0, 1].plot(self.training_history.history['loss'], label='Training Loss')
            axes[0, 1].plot(self.training_history.history['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot top-3 accuracy if available
            if 'top_3_accuracy' in self.training_history.history:
                axes[1, 0].plot(self.training_history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
                axes[1, 0].plot(self.training_history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
                axes[1, 0].set_title('Top-3 Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Top-3 Accuracy')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Plot learning rate if available
            if 'lr' in self.training_history.history:
                axes[1, 1].plot(self.training_history.history['lr'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.model_dir, 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
    
    def evaluate_model_detailed(self, X_test, y_test):
        """Perform detailed model evaluation"""
        logger.info("Performing detailed model evaluation...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_test)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Log results
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(f"Precision: {report['weighted avg']['precision']:.4f}")
        logger.info(f"Recall: {report['weighted avg']['recall']:.4f}")
        logger.info(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        # Plot confusion matrix
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            cm_path = os.path.join(self.model_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {cm_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model_components(self):
        """Save all model components"""
        # Save model
        model_path = os.path.join(self.model_dir, 'chatbot_v2_model.keras')
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to {encoder_path}")
        
        # Save training metadata
        metadata = {
            'max_sequence_length': self.max_sequence_length,
            'vocab_size': self.vocab_size,
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'word_index_size': len(self.tokenizer.word_index)
        }
        
        metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def train(self, epochs=100, batch_size=32):
        """CPU-optimized main training pipeline"""
        logger.info("="*60)
        logger.info("STARTING CPU-OPTIMIZED CHATBOT TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Prepare data
            patterns, labels = self.prepare_data()
            
            # Create tokenizer and encode data
            self.create_tokenizer(patterns)
            X = self.create_sequences(patterns)
            y = self.encode_labels(labels)
            
            # Build model
            num_classes = len(self.label_encoder.classes_)
            self.model = self.build_model(num_classes)
            
            # Train model
            history = self.train_model(X, y, epochs=epochs, batch_size=batch_size)
            
            # Plot training history
            self.plot_training_history()
            
            # Save everything
            self.save_model_components()
            
            # Log final summary
            logger.info("="*60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Best Accuracy: {self.best_accuracy:.4f}")
            logger.info(f"Training Time: {self.training_stats.get('training_time', 0):.2f} seconds")
            logger.info(f"Epochs Trained: {self.training_stats.get('epochs_trained', 0)}")
            logger.info(f"Model Parameters: {self.model.count_params():,}")
            logger.info("="*60)
            
            return history
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

def main():
    """Enhanced main training function"""
    try:
        # Initialize enhanced trainer
        trainer = EnhancedChatbotTrainer()
        
        # Start training with CPU-optimized parameters
        history = trainer.train(epochs=40, batch_size=32)
        
        # Print final results
        if trainer.training_stats:
            logger.info(f"Final Results Summary:")
            logger.info(f"  Best Accuracy: {trainer.training_stats['best_accuracy']:.4f}")
            logger.info(f"  Final Accuracy: {trainer.training_stats['final_accuracy']:.4f}")
            logger.info(f"  Training Time: {trainer.training_stats['training_time']:.2f} seconds")
            logger.info(f"  Epochs Trained: {trainer.training_stats['epochs_trained']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()