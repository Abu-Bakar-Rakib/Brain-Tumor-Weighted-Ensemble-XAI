# 🧠 Brain Tumor Weighted Ensemble XAI

**A Robust Weighted Ensemble Deep Learning Framework with Explainable AI for Cross-Dataset Multiclass Brain Tumor Classification on MRI**

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/Abu-Bakar-Rakib/Brain-Tumor-Weighted-Ensemble-XAI/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.8+-ff69b4.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org)

---

## 📌 Overview

This repository presents a comprehensive deep learning framework for brain tumor classification on MRI scans. It combines **six baseline CNN/transfer-learning architectures** with a novel **Dynamic Weighted Fusion Ensemble**.

**First public release** (`v1.0.0`) — all models trained and evaluated on Kaggle T4 x2, with full reproducibility.

---

## ✨ Key Features

- 🏗️ **6 Baseline Models**: CNN (from scratch), EfficientNetB0, MobileNetV3-Large, ResNet50, ConvNeXt-Tiny, Xception
- 🔗 **Novel Proposed Model**: 3-branch dynamic weighted fusion ensemble with learnable, softmax-normalized fusion weights
- 📊 **Comprehensive Evaluation**: Classification reports, confusion matrices, ROC-AUC (one-vs-rest + micro), loss/accuracy curves, calibration plots — all metrics to 4 decimal places
- 🎨 **4 XAI Methods**: Grad-CAM++, Score-CAM, Eigen-CAM, Attention Rollout (CNN-adapted)
- 🌍 **Cross-Dataset Validation**: External generalization testing on BRISC2025 (completely separate MRI dataset)
- ⚡ **Production-Ready**: Multi-GPU `MirroredStrategy`, mixed precision, optimized for Kaggle T4 x2

---

## 🏛️ Architecture

The ensemble combines three complementary branches with a learnable dynamic fusion layer:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT MRI IMAGE                             │
│                    (224×224×3, Preprocessed)                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   ResNet50    │ │EfficientNet-B5│ │  Custom CNN   │
│ (ImageNet-    │ │ (ImageNet-    │ │ (MRI-Optimized)
│  Pretrained)  │ │  Pretrained)  │ │               │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Feature Vec  │ │  Feature Vec  │ │  Feature Vec  │
│   (2048-dim)  │ │   (2048-dim)  │ │   (512-dim)   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │    DYNAMIC WEIGHTED FUSION      │
        │  (Learnable: w₁, w₂, w₃)        │
        │   Output = Σ wᵢ × Embeddingᵢ    │
        └─────────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │     FULLY CONNECTED LAYERS      │
        │      (512 → 256 → 128 → 4)      │
        └─────────────────┬───────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │       SOFTMAX OUTPUT            │
        │   [Glioma | Meningioma |        │
        │    Pituitary | No Tumor]        │
        └─────────────────────────────────┘
```

**Implementation Detail**: Each branch's feature vector is projected to a shared 512-dim embedding space via `Dense(512, relu)` before dynamic weighted fusion combines them.

---

## 📚 Datasets

| Role | Dataset | Size | Source |
|:---:|:---|:---|:---|
| **Train & In-Dist Test** | Brain Tumor MRI Dataset | 5,712 train / 1,311 test (4 classes) | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **External Validation** | BRISC2025 | 1,000 test images (unseen domain) | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025/) |

**Classes**: `glioma`, `meningioma`, `notumor` (mapped from BRISC2025's `no_tumor`), `pituitary`

---

## 📁 Repository Structure

```
brain-tumor-weighted-ensemble-xai/
├── README.md                            # This file
├── notebooks/
│   └── mri-brain.ipynb                  # Full end-to-end pipeline (Kaggle)
├── MRI
```

---

## 🚀 Quick Start (Kaggle)

### Step 1: Setup
1. Create a **new Kaggle notebook**
2. Enable **GPU: Tesla T4 x2** accelerator
3. Add datasets via **"Add Input"**:
   - `masoudnickparvar/brain-tumor-mri-dataset`
   - `briscdataset/brisc2025`

### Step 2: Run
4. Import or paste `notebooks/mri-brain.ipynb`
5. Execute from top to bottom:
   - Exploratory Data Analysis (EDA)
   - Baseline model training (6 models with early stopping, patience=7)
   - Proposed model training
   - XAI analysis
   - External validation on BRISC2025

### Step 3: Retrieve Results
6. All outputs saved to `/kaggle/working/outputs/`
7. Download from notebook's **Output** tab or attach `/kaggle/working` as a version

⏱️ **Runtime**: ~4–5 hours on T4 x2 (baselines dominate; EfficientNetB5 & proposed model are slowest)

---

## 📈 Results

### In-Distribution Test Set
**Brain Tumor MRI Dataset, 1,600 held-out test images**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) | ROC-AUC (micro) |
|---|---:|---:|---:|---:|---:|---:|
| CNN (from scratch) | 0.7769 | 0.7880 | 0.7769 | 0.7656 | 0.9327 | 0.9274 |
| EfficientNetB0 | 0.9319 | 0.9341 | 0.9319 | 0.9308 | 0.9906 | 0.9898 |
| MobileNetV3-Large | 0.9306 | 0.9339 | 0.9306 | 0.9300 | 0.9899 | 0.9897 |
| ResNet50 | 0.9444 | 0.9479 | 0.9444 | 0.9429 | 0.9912 | 0.9881 |
| ConvNeXt-Tiny | 0.9431 | 0.9468 | 0.9431 | 0.9419 | 0.9896 | 0.9877 |
| Xception | 0.9475 | 0.9503 | 0.9475 | 0.9463 | 0.9920 | 0.9889 |
| **🔗 Proposed (Weighted Fusion)** | 0.9494 | 0.9528 | 0.9494 | 0.9481 | **0.9940** | **0.9902** |

### Cross-Dataset Validation
**BRISC2025, 1,000 fully unseen images**

| Model | Accuracy | F1 (macro) | ROC-AUC (macro) |
|---|---:|---:|---:|
| CNN (from scratch) | 0.742 | 0.7227 | 0.9244 |
| EfficientNetB0 | 0.949 | 0.9458 | 0.9962 |
| MobileNetV3-Large | 0.952 | 0.9482 | 0.9953 |
| ResNet50 | 0.967 | 0.9608 | 0.9988 |
| ConvNeXt-Tiny | 0.951 | 0.9377 | 0.9946 |
| Xception | 0.980 | 0.9803 | 0.9991 |
| **🔗 Proposed (Weighted Fusion)** | **0.982** | **0.9823** | 0.9978 |

### Key Findings

✅ **In-distribution**: The proposed model achieves strong ROC-AUC (0.9940 vs 0.9902 micro), demonstrating excellent discrimination capability across all classes

✅ **Cross-dataset**: Proposed model achieves **best accuracy (0.982)** and **best F1 (0.9823)** on BRISC2025, demonstrating superior generalization

✅ **Learned Fusion Weights** (softmax-normalized):
- ResNet50: 0.3392
- EfficientNetB5: 0.3334
- Custom CNN: 0.3274

→ Nearly **equal contribution** from all branches, indicating complementary, non-redundant feature learning

---

## 🎨 Explainable AI (XAI)

Four XAI methods applied to the ResNet50 branch's intermediate layers:

| Method | Description |
|---|---|
| **Grad-CAM++** | Gradient-weighted class activation mapping (improved weighting) |
| **Score-CAM** | Score-based CAM (gradient-free, more stable) |
| **Eigen-CAM** | Eigenvector-weighted activation mapping |
| **Attention Rollout** | CNN-adapted transformer attention mechanism aggregation |

Visualizations focus on ResNet50's conv blocks: `conv3_block4_out`, `conv4_block6_out`, `conv5_block3_out`

> **Note**: Attention Rollout adapted for CNNs following standard literature (cumulative attention across layers)

---

## 📋 Requirements

```
TensorFlow/Keras        >= 2.15
opencv-python          
scikit-learn           
pandas                 
matplotlib             
seaborn                
```

**Hardware**: Kaggle T4 x2 GPU recommended (auto-detects & uses `MirroredStrategy` on multi-GPU setups; single-GPU compatible)

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 📝 Changelog

### v1.0.0 (Initial Release)
- ✅ 6 baseline models (CNN, EfficientNetB0, MobileNetV3-Large, ResNet50, ConvNeXt-Tiny, Xception)
- ✅ Dynamic weighted fusion ensemble (ResNet50 + EfficientNetB5 + Custom CNN)
- ✅ Comprehensive evaluation suite per model
- ✅ 4 XAI methods on proposed model
- ✅ External cross-dataset validation on BRISC2025
- ✅ Full reproducibility on Kaggle T4 x2

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Contact & Support

For questions, issues, or collaborations:
- 📍 GitHub Issues: [Open an issue](https://github.com/Abu-Bakar-Rakib/Brain-Tumor-Weighted-Ensemble-XAI/issues)
- 👤 Author: [Abu-Bakar-Rakib](https://github.com/Abu-Bakar-Rakib)

---

<div align="center">

**⭐ If you found this helpful, please consider giving it a star!**

Made with ❤️ for advancing medical AI

</div>
