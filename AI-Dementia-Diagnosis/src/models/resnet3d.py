import tensorflow as tf
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger(__name__)

def build_resnet3d_model(input_shape=(64, 64, 64, 1), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling3D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output (binary classification)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ResNet3D')
    logger.info(f"Model built: {model.count_params():,} parameters")
    
    return model