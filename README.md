# AI-Dementia-Diagnosis

Deep learning for early dementia detection from 3D MRI brain scans.

## Overview

Binary classification system using 3D CNN to detect dementia from structural MRI data (OASIS-2 dataset).

**Dataset:** 373 MRI scans (150 subjects, longitudinal data)  
**Performance:** 58.9% accuracy, 0.55 AUC on test set

## Setup
```bash
conda create -n dementia-diagnosis python=3.10
conda activate dementia-diagnosis
pip install -r requirements.txt
```

## Data

Download OASIS-2 dataset:
1. Visit https://www.oasis-brains.org
2. Extract to `data/raw/OAS2_RAW_PART1` and `OAS2_RAW_PART2`
3. Download demographics: `data/raw/oasis_longitudinal_demographics.xlsx`

## Usage

**Preprocess:**
```bash
python scripts/preprocess_oasis2.py
```

**Train:**
```bash
python scripts/train.py
```

**Predict:**
```bash
python scripts/predict.py --input path/to/scan.nii
```

## Model

- Architecture: 3D CNN (3 conv blocks + dense layers)
- Input: 64×64×64×1 MRI volume
- Output: Binary classification (dementia/normal)
- Parameters: ~345K trainable

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 58.9% |
| AUC | 0.55 |
| Precision | 55.0% |
| Recall | 44.0% |

*Limited by small dataset size (373 samples)*

## Structure
```
├── src/
│   ├── preprocessing/mri_processor.py
│   ├── models/resnet3d.py
│   ├── training/trainer.py
│   └── utils/logger.py
├── scripts/
│   ├── preprocess_oasis2.py
│   ├── train.py
│   └── predict.py
└── config.yaml
```

## Limitations

- Small dataset (risk of overfitting)
- Single-site data (may not generalize)
- Binary classification only

## Contact

**Georgii Erokhin**  
[GitHub](https://github.com/t0gae) | [Email](mailto:georgii.erokhin@gmail.com)
