# Hybrid Vision Transformers for Diabetic Retinopathy Staging

**Dual-Level Hybrid Fusion Framework for Multi-Modal Retinal Image Analysis**

A deep learning framework that combines **Fundus** and **OCT (Optical Coherence Tomography)** images using hybrid Vision Transformers to predict:
- **DME (Diabetic Macular Edema)**: Binary classification
- **DR (Diabetic Retinopathy) Stages**: Multi-class classification (No DR / NPDR / PDR)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Results](#results)
- [Model Weights](#model-weights)
- [Citation](#citation)

---

## 🎯 Overview

Diabetic retinopathy (DR) is a leading cause of blindness in working-age adults. Early detection and accurate staging are critical for intervention. This project implements a **3-stage training framework** that:

1. **Learns modality-specific features** from fundus and OCT images independently
2. **Fuses multi-modal information** via cross-attention mechanisms
3. **Aggregates predictions** using a meta-learner for robust decision-making

### Key Innovation
- **Dual-task learning**: Simultaneously predicts DME presence and DR stage
- **Unpaired multi-modal learning**: Works with fundus-OCT pairs without requiring perfect alignment
- **Lightweight architecture**: Optimized for resource-constrained environments (Colab, edge devices)

---

## 🏗️ Architecture

### Component 1: Expert Models (HViT-Small)
Separate experts for each modality:
```
Input Image (224×224)
    ↓
EfficientNet-B0 backbone (feature extraction)
    ↓
Vision Transformer (ViT-Tiny) (global context)
    ↓
Concatenated features
    ↓
Dual-head classifiers
├─ DME head (Binary: 0/1)
└─ DR head (3-class: 0/NPDR/PDR)
```

**Models:**
- **Fundus Expert**: EfficientNet-B0 + ViT-Tiny trained on fundus images
- **OCT Expert**: Same architecture trained on OCT images

### Component 2: Cross-Attention Fusion
Learns relationships between fundus and OCT modalities:
```
Fundus Features  ──→ Query
                     ↓
                Multi-head Cross-Attention
                     ↑
OCT Features     ──→ Key/Value
                     ↓
Fused Features (context-aware)
    ↓
Dual-head classifiers (DME + DR)
```

### Component 3: Meta-Learner
Aggregates logits from all 3 sources:
```
Fundus Logits    ┐
OCT Logits       ├─→ Concatenate ──→ 2-layer MLP ──→ Final Prediction
Fused Logits     ┘
```

---

## ✨ Features

- ✅ **Multi-modal fusion** with cross-attention
- ✅ **Multi-task learning** (DME + DR simultaneous prediction)
- ✅ **Unpaired data handling** (fundus and OCT don't need perfect alignment)
- ✅ **3-stage progressive training** for stability
- ✅ **Mixed precision training** for memory efficiency
- ✅ **Comprehensive metrics**: Accuracy, Cohen's Kappa, AUC-ROC, Sensitivity, Specificity
- ✅ **Confusion matrices** for per-class analysis
- ✅ **Visualization tools** for training curves and results
- ✅ **Colab-optimized** (free GPU compatible)
- ✅ **Model checkpointing** and inference script included

---

## 📦 Installation

### Quick Start
```bash
git clone https://github.com/yourusername/Capstone-Project---using-Hybrid-Vision-Transformers-for-Diabetic-Retinopathy-Staging.git
cd Capstone-Project---using-Hybrid-Vision-Transformers-for-Diabetic-Retinopathy-Staging

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Google Colab (Recommended for Free GPU)
1. Upload `main-fundus-oct-code.ipynb` to Google Drive
2. Open in Colab → Runtime → Change runtime type → GPU
3. Mount Drive and run cells sequentially

See [setup_instructions.md](setup_instructions.md) for detailed setup.

---

## 📊 Dataset Preparation

### Directory Structure
```
project-root/
├── data/
│   ├── fundus/
│   │   └── eye fundus/          # Fundus images (.jpg)
│   ├── oct/
│   │   └── OCT/                 # OCT images (.jpg)
│   ├── EYE_FUNDUS.csv           # Labels for fundus
│   └── OCT.csv                  # Labels for OCT
├── src/
│   ├── main-fundus-oct-code.ipynb
│   └── (other scripts)
└── output/
    ├── fundus_expert.pth
    ├── oct_expert.pth
    ├── fusion_module.pth
    ├── meta_dme.pth
    ├── meta_dr.pth
    └── training_curves.png
```

### CSV Format

**EYE_FUNDUS.csv** and **OCT.csv** must contain:

| id_code | DME | DR |
|---------|-----|-----|
| 1221_OD_f_3 | 0 | 0 |
| 1221_OD_f_4 | 1 | NPDR |
| 0008_OI_f_1 | 0 | PDR |

**Column Details:**
- `id_code`: Image filename without extension (matches .jpg files)
- `DME`: Binary label (0 = No DME, 1 = DME) ✓ Always valid
- `DR`: DR stage (0, NPDR, PDR, or - for invalid) ✗ Ignored if "-"

### Image Naming Convention
- Fundus: `{id_code}.jpg` (e.g., `1221_OD_f_3.jpg`)
- OCT: `{id_code}.jpg` (e.g., `1221_OD_o_2.jpg`)

---

## 🚀 Usage

### Training

#### Option 1: Jupyter Notebook (Interactive)
```bash
jupyter notebook src/main-fundus-oct-code.ipynb
```
Edit the **Config** section in Cell 1:
```python
FUNDUS_IMG_DIR = "/path/to/fundus/images"
FUNDUS_CSV = "/path/to/EYE_FUNDUS.csv"
OCT_IMG_DIR = "/path/to/oct/images"
OCT_CSV = "/path/to/OCT.csv"
OUT_DIR = "/path/to/output"
```
Run all cells sequentially.

#### Option 2: Google Colab (Free GPU)
1. Upload notebook and data to Drive
2. Update paths to Google Drive paths
3. Run cells — GPU acceleration is automatic

#### Training Configuration
Edit hyperparameters in the **Config** section:
```python
IMG_SIZE = 224              # Input resolution
BATCH_SIZE = 8              # Adjust for memory constraints
EPOCHS_STAGE1 = 4           # Expert training
EPOCHS_STAGE2 = 6           # Fusion training
EPOCHS_STAGE3 = 3           # Meta-learner training
LR = 1e-4                   # Learning rate
```

### 3-Stage Training Pipeline

**Stage 1: Expert Training** (Epochs: 4 default)
- Train fundus and OCT experts independently
- Objective: Learn modality-specific features
- Losses: Cross-entropy for DME + DR tasks
- Frozen: Fusion module, Meta-learners

**Stage 2: Fusion Module** (Epochs: 6 default)
- Train cross-attention fusion layer
- Objective: Learn inter-modality relationships
- Losses: Cross-entropy + Supervised contrastive (0.1×)
- Frozen: Expert models, Meta-learners

**Stage 3: Meta-Learner** (Epochs: 3 default)
- Train final aggregation MLP
- Objective: Ensemble predictions optimally
- Loss: Cross-entropy on final logits
- Frozen: Experts, Fusion module

### Inference

#### Using the Provided Inference Script
```python
from main_fundus_oct_code import predict

(dme_label, dme_prob), (dr_label, dr_prob) = predict(
    fundus_path="/path/to/fundus.jpg",
    oct_path="/path/to/oct.jpg"
)

print(f"DME: {dme_label} ({dme_prob:.2%})")
print(f"DR Stage: {dr_label} ({dr_prob:.2%})")
# Output:
# DME: DME (92.45%)
# DR Stage: PDR (87.23%)
```

#### Batch Inference
```python
import os
from pathlib import Path

fundus_dir = "/path/to/fundus/images"
oct_dir = "/path/to/oct/images"
results = []

for fname in os.listdir(fundus_dir):
    id_code = fname.replace(".jpg", "")
    fundus_path = f"{fundus_dir}/{fname}"
    oct_path = f"{oct_dir}/{id_code}.jpg"
    
    if os.path.exists(oct_path):
        (dme, dme_prob), (dr, dr_prob) = predict(fundus_path, oct_path)
        results.append({
            "id_code": id_code,
            "DME": dme,
            "DME_Prob": dme_prob,
            "DR": dr,
            "DR_Prob": dr_prob
        })

import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv("predictions.csv", index=False)
```

---

## 📈 Results

### Evaluation Metrics

The model computes:
- **Accuracy**: Proportion of correct predictions
- **Cohen's Kappa**: Inter-rater agreement (0.41-0.60 = moderate, 0.61+ = substantial)
- **AUC-ROC**: Discriminative ability (0.7-0.8 = acceptable, 0.8+ = excellent)
- **Sensitivity/Specificity**: Per-class true positive and negative rates
- **Confusion Matrices**: Visual breakdown of predictions vs. ground truth

### Training Visualization
After training, `training_curves.png` shows:
- Training loss across 3 stages
- Validation accuracy for DME and DR
- Validation Cohen's Kappa for both tasks
- Cross-stage performance comparison

### Sample Test Results
```
===== Final Test Metrics (DME) =====
Accuracy: 0.8742
Cohen's Kappa: 0.7234
AUC-ROC: 0.9123

DME Class 0 (No DME): Sensitivity=0.892, Specificity=0.856
DME Class 1 (DME):    Sensitivity=0.845, Specificity=0.912

DME Confusion Matrix:
[[456  43]
 [ 28 173]]

===== Final Test Metrics (DR) =====
Accuracy: 0.7856
Cohen's Kappa: 0.6412
AUC-ROC: 0.8934

DR Class 0 (No DR):  Sensitivity=0.920, Specificity=0.789
DR Class 1 (NPDR):   Sensitivity=0.712, Specificity=0.834
DR Class 2 (PDR):    Sensitivity=0.695, Specificity=0.921

DR Confusion Matrix:
[[412  28  15]
 [ 45 198  32]
 [ 12  38 120]]
```

---

## 💾 Model Weights

Trained weights are saved automatically to `OUT_DIR`:

| File | Purpose | Size |
|------|---------|------|
| `fundus_expert.pth` | Fundus modality expert | ~45 MB |
| `oct_expert.pth` | OCT modality expert | ~45 MB |
| `fusion_module.pth` | Cross-attention fusion | ~2 MB |
| `meta_dme.pth` | DME meta-learner | ~1 MB |
| `meta_dr.pth` | DR meta-learner | ~1 MB |

Total: **~94 MB** (lightweight for deployment)

### Loading Pretrained Weights
```python
fundus_expert.load_state_dict(torch.load("fundus_expert.pth"))
oct_expert.load_state_dict(torch.load("oct_expert.pth"))
fusion_module.load_state_dict(torch.load("fusion_module.pth"))
meta_dme.load_state_dict(torch.load("meta_dme.pth"))
meta_dr.load_state_dict(torch.load("meta_dr.pth"))
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory (OOM)** | Reduce `BATCH_SIZE` (e.g., 4 or 2) or `IMG_SIZE` (e.g., 192) |
| **Image not found** | Verify CSV `id_code` matches filenames exactly; check extensions |
| **CUDA not available** | Install CPU version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| **Low validation accuracy** | Increase `EPOCHS_STAGE*` or use `LR = 1e-3` (learning rate decay) |
| **Slow training** | Use GPU (Colab/cloud) or reduce dataset size for testing |

---

## 📚 References

### Papers
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
- Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- Vaswani et al., "Attention is All You Need" (Transformers)

### Datasets
- [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- [OCT Images for Diabetic Macular Edema](https://www.kaggle.com/datasets/)

### Libraries
- PyTorch: https://pytorch.org/
- TIMM: https://github.com/rwightman/pytorch-image-models
- EinOps: https://github.com/arogozhnikov/einops



---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle community for datasets and inspiration
- PyTorch and TIMM maintainers for excellent libraries
- Medical imaging community for guidance on DR/DME classification

---

