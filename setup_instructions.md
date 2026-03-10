# Setup Instructions - Hybrid Vision Transformers for Diabetic Retinopathy Staging

## Overview
This project implements a **Dual-Level Hybrid Fusion Framework** combining Fundus and OCT images to predict:
- **DME (Diabetic Macular Edema)**: Binary classification (No DME / DME)
- **DR (Diabetic Retinopathy) Stages**: Multi-class classification (No DR / NPDR / PDR)

## Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)
- 8GB+ RAM
- 20GB+ storage for datasets

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Capstone-Project---using-Hybrid-Vision-Transformers-for-Diabetic-Retinopathy-Staging.git
cd Capstone-Project---using-Hybrid-Vision-Transformers-for-Diabetic-Retinopathy-Staging
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dr-staging python=3.10
conda activate dr-staging
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages:**
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Computer vision utilities
- `timm>=0.9.0` - Timely Models library (Vision Transformers)
- `einops>=0.7.0` - Tensor operations
- `scikit-learn>=1.3.0` - ML metrics & preprocessing
- `pandas>=1.5.0` - Data manipulation
- `Pillow>=9.0.0` - Image processing
- `matplotlib>=3.7.0` - Visualization
- `numpy>=1.23.0` - Numerical computing

## Data Preparation

### Directory Structure
```
project-root/
├── data/
│   ├── fundus/
│   │   └── eye fundus/          # Fundus images (.jpg)
│   ├── oct/
│   │   └── OCT/                 # OCT images (.jpg)
│   ├── EYE_FUNDUS.csv           # Fundus labels
│   └── OCT.csv                  # OCT labels
├── src/
│   └── main-fundus-oct-code.ipynb
└── output/
    └── (weights and results)
```

### CSV Format Requirements

**EYE_FUNDUS.csv** and **OCT.csv** must contain:
```
id_code,DME,DR
1221_OD_f_3,0,0
1221_OD_f_4,1,NPDR
0008_OI_f_1,0,PDR
...
```

**Columns:**
- `id_code`: Image filename without extension (must match .jpg files)
- `DME`: Binary label (0=No DME, 1=DME)
- `DR`: DR stage (0=No DR, NPDR=Non-proliferative, PDR=Proliferative, -=Invalid/Unknown)

### Image Naming Convention
- Fundus: `{id_code}.jpg` (e.g., `1221_OD_f_3.jpg`)
- OCT: `{id_code}.jpg` (e.g., `1221_OD_o_2.jpg`)

## Running the Code

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook src/main-fundus-oct-code.ipynb
```

**Steps:**
1. Update paths in the **Config** section (cell 1):
   ```python
   FUNDUS_IMG_DIR = "/path/to/eye fundus"
   FUNDUS_CSV = "/path/to/EYE_FUNDUS.csv"
   OCT_IMG_DIR = "/path/to/OCT"
   OCT_CSV = "/path/to/OCT.csv"
   OUT_DIR = "/path/to/output"
   ```
2. Run all cells sequentially
3. Monitor training across 3 stages in the console

### Option 2: Google Colab (Free GPU)
1. Upload notebook to Google Drive
2. Open in Colab
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
   ```
4. Upload CSVs and images to Drive
5. Update paths and run

## Configuration Parameters

Edit in the notebook's **Config** section:

```python
# Image & batch settings
IMG_SIZE = 224              # Input image size
BATCH_SIZE = 8              # Batch size (reduce if OOM)
EPOCHS_STAGE1 = 4           # Expert training epochs
EPOCHS_STAGE2 = 6           # Fusion module epochs
EPOCHS_STAGE3 = 3           # Meta-learner epochs
LR = 1e-4                   # Learning rate

# Data split
val_ratio = 0.12            # 12% validation
test_ratio = 0.08           # 8% test
SEED = 42                   # Reproducibility
```

## Training Pipeline

### 3-Stage Training Strategy

**Stage 1: Expert Training** (4 epochs default)
- Train EfficientNet-B0 + ViT-Tiny experts on fundus and OCT separately
- Optimize both DME and DR classifiers
- Loss: Cross-entropy on both tasks

**Stage 2: Fusion Module** (6 epochs default)
- Train cross-attention fusion between modalities
- Use supervised contrastive learning on feature pairs
- Loss: CE + Contrastive (0.1× weight)

**Stage 3: Meta-Learner** (3 epochs default)
- Aggregate logits from all 3 sources (fundus, OCT, fused)
- Train final decision MLP
- Loss: Cross-entropy on final predictions

### Monitoring
- Training loss printed every 20 steps
- Validation metrics (Accuracy, Cohen's Kappa, AUC) after each epoch
- Training curves saved as `training_curves.png`

## Output Files

After training, the following files are saved to `OUT_DIR`:
```
output/
├── fundus_expert.pth        # Fundus modality expert
├── oct_expert.pth           # OCT modality expert
├── fusion_module.pth        # Cross-attention fusion
├── meta_dme.pth             # DME meta-learner
├── meta_dr.pth              # DR meta-learner
└── training_curves.png      # Loss & metrics plots
```

## Inference

Use the provided inference script to make predictions on new image pairs:

```python
from main-fundus-oct-code import predict

# Predict on a fundus-OCT pair
(dme_pred, dme_prob), (dr_pred, dr_prob) = predict(
    fundus_path="path/to/fundus.jpg",
    oct_path="path/to/oct.jpg"
)

print(f"DME: {dme_pred} ({dme_prob:.2%})")
print(f"DR Stage: {dr_pred} ({dr_prob:.2%})")
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` (e.g., 4 or 2)
- Reduce `IMG_SIZE` (e.g., 192 or 160)
- Use GPU with higher memory or CPU (slower)

### Image Not Found Errors
- Verify CSV `id_code` matches image filenames exactly
- Check file extensions (.jpg vs .png)
- Use absolute paths or verify relative paths from working directory

### CUDA Not Available
- Install CPU version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Training will be slower but functional

### Mismatched Label Classes
- Ensure DR labels are exactly: `0`, `NPDR`, `PDR`, or `-`
- Invalid labels (marked `-`) are automatically filtered during training

## Performance Metrics

The model evaluates:
- **Accuracy**: Proportion of correct predictions
- **Cohen's Kappa**: Inter-rater agreement (accounts for chance)
- **AUC-ROC**: Area under receiver operating curve
- **Sensitivity & Specificity**: Per-class performance
- **Confusion Matrix**: Per-class true/false positives

## Reference

**Architecture:**
- Backbone: EfficientNet-B0 + Vision Transformer (ViT-Tiny)
- Fusion: Multi-head Cross-Attention
- Meta-learner: 2-layer MLP

**Paper:** Hybrid Vision Transformers for Medical Image Analysis

## Contact & Support
For issues or questions, open a GitHub issue or contact the project maintainers.