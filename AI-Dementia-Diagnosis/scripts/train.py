#!/usr/bin/env python3
import sys
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))
from src.models.resnet3d import build_resnet3d_model
from src.training.trainer import DementiaTrainer
from src.utils.logger import setup_logger

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    processed_dir = Path(config['data']['processed_dir'])
    X = np.load(processed_dir / 'mri_scans.npy')
    y = np.load(processed_dir / 'labels.npy')
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Fix dimensions if needed
    if X.ndim == 6:
        X = X.squeeze(axis=-1)
        print(f"Fixed shape: {X.shape}")
    
    return X, y

def split_data(X, y, config):
    train_size = config['data']['train_split']
    val_size = config['data']['val_split']
    test_size = config['data']['test_split']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=42
    )
    
    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} (Normal: {sum(y_train==0)}, Dementia: {sum(y_train==1)})")
    print(f"  Val: {len(X_val)} (Normal: {sum(y_val==0)}, Dementia: {sum(y_val==1)})")
    print(f"  Test: {len(X_test)} (Normal: {sum(y_test==0)}, Dementia: {sum(y_test==1)})")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main():
    logger = setup_logger('train', 'logs/train.log')
    config = load_config()
    
    logger.info("Loading data")
    X, y = load_data(config)
    
    train_data, val_data, test_data = split_data(X, y, config)
    
    logger.info("Building model")
    model = build_resnet3d_model(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes']
    )
    model.summary()
    
    trainer = DementiaTrainer(model, config)
    trainer.compile_model()
    
    logger.info("Training")
    trainer.train(train_data, val_data)
    
    logger.info("Evaluating")
    metrics = trainer.evaluate(test_data)
    
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

if __name__ == '__main__':
    main()