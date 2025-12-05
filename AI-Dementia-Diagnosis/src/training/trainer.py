import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

class DementiaTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = None
    
    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['model']['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        logger.info("Model compiled")
    
    def get_callbacks(self):
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),  
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    def compute_class_weights(self, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        return dict(zip(classes, weights))
    
    def train(self, train_data: Tuple, val_data: Tuple):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        class_weights = self.compute_class_weights(y_train)
        logger.info(f"Class weights: {class_weights}")
        
        logger.info(f"Training on {len(X_train)} samples")
        logger.info(f"Validation on {len(X_val)} samples")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        logger.info("Training complete")
        return self.history
    
    def evaluate(self, test_data: Tuple):
        X_test, y_test = test_data
        results = self.model.evaluate(X_test, y_test, verbose=1)
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info(f"Test metrics: {metrics}")
        return metrics