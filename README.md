#  AI-Driven Multimodal Diagnosis of Dementia  


##  Abstract  
**Problem:** 60% of dementia cases in Japan are diagnosed at advanced stages ([Ministry of Health, 2023](https://www.mhlw.go.jp/)).  

**Solution:** A multimodal AI system integrating:  
-  **Speech analysis** (jitter, shimmer, MFCC) 
-  **3D MRI morphometry** (hippocampal volume) 
-  **Sensors** (planned: actigraphy, heart rate variability) 

**Innovations:**  
1. First **cross-modal attention** model for speech-MRI fusion
2. Planned **edge deployment** on Raspberry Pi 5  

---

##  Methodology  
### Model Architecture  

**Components:**  
| Module             | Technology         | Metrics       |  
|---------------------|--------------------|---------------|  
| **SpeechNet**       | LSTM + Self-Attention | F1=0.82     |  
| **NeuroImageNet**   | 3D ResNet-50       | AUC=0.89      |  
| **Fusion Layer**    | Transformer        | AUC=0.93      |  

**Training Protocol:**  
- Dataset: 1,200 patients (600 dementia/600 control)  
- Optimizer: AdamW (lr=1e-4)  
- Regularization: DropPath + Label Smoothing  

---

##  Results  
### Key Metrics  
| Parameter         | Value    | 95% ДИ       |  
|-------------------|----------|--------------|  
| **AUC-ROC**       | 0.93     | [0.91–0.95]  |  
| **Sensitivity** | 0.87   | [0.84–0.90]  |  
| **Specificity**   | 0.89   | [0.86–0.92]  |  


---

##  Social Impact  
 

**Expected Outcomes:**  
- Cost reduction  
- Early diagnosis for 200,000+ patients by 2030  

---

##  Development Roadmap  
1.  Integration of wearable device data (Fitbit, Apple Watch) 
2.  Federated learning implementation for privacy preservation  
3.  Clinical trials under **J-ADNI** protocol

---

##  Installation & Usage 
```bash
# 1. Clone Repository
git clone https://github.com/t0gae/AI-Dementia-Diagnosis-MEXT
cd AI-Dementia-Diagnosis

# 2. Run Diagnosis
python diagnose.py \
  --audio data/input/patient.wav \
  --mri data/input/brain.nii.gz \
  --model models/best_model.h5 \
  --output reports/diagnosis.json
