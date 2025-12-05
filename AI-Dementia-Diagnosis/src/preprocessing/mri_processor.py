import nibabel as nib
import numpy as np
from skimage.transform import resize
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MRIPreprocessor:
    def __init__(self, target_shape=(64, 64, 64), normalize=True):
        self.target_shape = target_shape
        self.normalize = normalize
    
    def load_mri(self, file_path):
        try:
            img = nib.load(file_path)
            return img.get_fdata()
        except Exception as e:
            logger.error(f"Load failed {file_path}: {e}")
            raise
    
    def normalize_intensity(self, data):
        if not self.normalize:
            return data
        
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            return np.zeros_like(data)
        
        return (data - data_min) / (data_max - data_min)
    
    def resize_volume(self, data):
        if data.shape == self.target_shape:
            return data
        return resize(data, self.target_shape, mode='constant', anti_aliasing=True, preserve_range=True)
    
    def preprocess(self, file_path):
        data = self.load_mri(file_path)
        data = self.normalize_intensity(data)
        data = self.resize_volume(data)
        data = np.expand_dims(data, axis=-1)
        return data