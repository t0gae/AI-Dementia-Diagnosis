import sys
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.mri_processor import MRIPreprocessor
from src.utils.logger import setup_logger

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict dementia from MRI scan')
    parser.add_argument('--input', required=True, help='Path to MRI scan (.nii file)')
    parser.add_argument('--model', help='Path to trained model (default: from config)')
    args = parser.parse_args()
    
    logger = setup_logger('predict', 'logs/predict.log')
    config = load_config()
    
    model_path = args.model or config['inference']['model_path']
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    logger.info(f"Processing {args.input}")
    preprocessor = MRIPreprocessor(
        target_shape=tuple(config['preprocessing']['target_shape']),
        normalize=config['preprocessing']['normalize']
    )
    
    mri_data = preprocessor.preprocess(args.input)
    mri_data = np.expand_dims(mri_data, axis=0)
    
    prediction = model.predict(mri_data, verbose=0)[0][0]
    threshold = config['inference']['threshold']
    
    result = "Dementia detected" if prediction >= threshold else "No dementia detected"
    confidence = prediction if prediction >= threshold else 1 - prediction
    
    logger.info(f"Prediction: {result}")
    logger.info(f"Confidence: {confidence:.2%}")
    
    print(f"\nResult: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw score: {prediction:.4f}")

if __name__ == '__main__':
    main()