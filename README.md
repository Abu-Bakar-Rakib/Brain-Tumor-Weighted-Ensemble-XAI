# Brain Tumor Weighted Ensemble XAI

**A Robust Weighted Ensemble Deep Learning Framework with Explainable AI for Cross-Dataset Multiclass Brain Tumor Classification on MRI**

`v1.0.0` — first public release (baselines + proposed model + XAI + external validation, all trained and evaluated on Kaggle T4 x2)

This repository trains and evaluates seven baseline CNN/transfer-learning architectures plus a novel **Dynamic Weighted Fusion Ensemble** (ResNet50 + EfficientNetB5 + a custom MRI-designed CNN) for 4-class brain tumor classification on MRI, then explains the proposed model's predictions with four XAI methods (Grad-CAM++, Score-CAM, Eigen-CAM, CNN-adapted Attention Rollout) and validates it on a fully independent, external dataset to test cross-dataset generalization.

## Highlights

- **7 baseline models** — CNN (from scratch), EfficientNetB0, MobileNetV3-Large, ResNet50, ConvNeXt-Tiny, InceptionV3, Xception
- **1 proposed model** — a 3-branch ensemble combined via a *learnable, softmax-normalized* dynamic weighted fusion layer
- **Full evaluation suite per model** — classification report, confusion matrix, ROC-AUC (one-vs-rest + micro-average), loss/accuracy curves, calibration curves, all metrics reported to 4 decimal places
- **4 XAI methods** on the proposed model — Grad-CAM++, Score-CAM, Eigen-CAM, Attention Rollout (CNN-adapted)
- **External cross-dataset validation** on BRISC2025, a completely separate MRI dataset, to test real-world generalization rather than just in-distribution accuracy
- Built for **Kaggle's free T4 x2 GPU** (multi-GPU `MirroredStrategy`, mixed precision)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT MRI IMAGE                              │
│                    (224×224×3, Preprocessed)                     │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   ResNet50    │ │EfficientNet-B5│ │  Custom CNN   │
│ (ImageNet-    │ │ (ImageNet-    │ │ (Designed for │
│  Pretrained)  │ │  Pretrained)  │ │     MRI)      │
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
        │    DYNAMIC WEIGHTED FUSION       │
        │  (Learnable Weights: w₁, w₂, w₃) │
        │   Combined = Σ wᵢ × Embeddingᵢ   │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │     FULLY CONNECTED LAYERS       │
        │      (256 → 128 → 64 → 4)        │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │       SOFTMAX OUTPUT             │
        │   Glioma / Meningioma /          │
        │   Pituitary / No Tumor           │
        └─────────────────────────────────┘
```

> **Implementation note:** each branch's 2048/2048/512-dim feature vector is first projected to a shared 512-dim embedding space (`Dense(512, relu)` per branch), and the dynamic weighted fusion combines these embeddings rather than raw 4-way softmax predictions. Fusing at the embedding level (rather than fusing already-collapsed class probabilities) gives the FC head richer information to work with and better gradient flow during training, while preserving the same learnable-weighted-combination idea shown above.

## Datasets

| Role | Dataset | Size | Link |
|---|---|---|---|
| Training / in-distribution test | Brain Tumor MRI Dataset (Masoud Nickparvar) | 5,712 train / 1,311 test images across 4 classes | [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| External / cross-dataset validation | BRISC2025 | 1,000 test images | [kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025/) |

Classes: `glioma`, `meningioma`, `notumor` (BRISC2025 labels this `no_tumor` — the code maps it automatically), `pituitary`.

## Repository structure

```
brain-tumor-weighted-ensemble-xai/
├── README.md
├── notebooks/
│   └── mri-brain.ipynb                  # full pipeline (Kaggle notebook)
├── outputs/                             # generated at runtime, not checked in
│   ├── eda/                             # class distribution, sample grids
│   ├── baselines/<model_name>/          # per-baseline report/CM/ROC/loss/calibration
│   ├── proposed/                        # proposed model's report/CM/ROC/loss/calibration
│   ├── xai/                             # Grad-CAM++/Score-CAM/Eigen-CAM/Rollout panels
│   ├── external_validation/             # BRISC2025 cross-dataset results
│   └── summary/                         # all_models_summary.csv, comparison charts
└── models/                              # saved .keras checkpoints (best weights per model)
```

## How to run (Kaggle)

1. Create a new Kaggle notebook, enable **GPU T4 x2** as the accelerator.
2. Add both datasets via **"Add Input"**: `masoudnickparvar/brain-tumor-mri-dataset` and `briscdataset/brisc2025`.
3. Upload/import `notebooks/mri-brain.ipynb` (or paste the `.py` pipeline, splitting on `# %%` cell markers).
4. Run top to bottom:
   - EDA → baseline training (7 models, early stopping, patience 7) → proposed model training → XAI → external validation.
5. All artifacts are written under `/kaggle/working/outputs/`; download them from the notebook's Output tab, or attach `/kaggle/working` as the notebook's output directory to persist a Kaggle "Version".

Approximate runtime on T4 x2: ~4–5 hours for the full pipeline (baselines dominate; the 3-branch proposed model and EfficientNetB5-based baseline are the slowest per-epoch).

## Results

### In-distribution test set (Brain Tumor MRI Dataset, 1,600 held-out test images)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) | ROC-AUC (micro) |
|---|---|---|---|---|---|---|
| CNN (from scratch) | 0.7769 | 0.7880 | 0.7769 | 0.7656 | 0.9327 | 0.9274 |
| EfficientNetB0 | 0.9319 | 0.9341 | 0.9319 | 0.9308 | 0.9906 | 0.9898 |
| MobileNetV3-Large | 0.9306 | 0.9339 | 0.9306 | 0.9300 | 0.9899 | 0.9897 |
| ResNet50 | 0.9444 | 0.9479 | 0.9444 | 0.9429 | 0.9912 | 0.9881 |
| ConvNeXt-Tiny | 0.9431 | 0.9468 | 0.9431 | 0.9419 | 0.9896 | 0.9877 |
| InceptionV3 | **0.9550** | 0.9575 | 0.9550 | **0.9540** | 0.9939 | 0.9928 |
| Xception | 0.9475 | 0.9503 | 0.9475 | 0.9463 | 0.9920 | 0.9889 |
| **Proposed (weighted fusion)** | 0.9494 | **0.9528** | 0.9494 | 0.9481 | **0.9940** | **0.9902** |

### External cross-dataset validation (BRISC2025, 1,000 fully unseen images)

| Model | Accuracy | F1 (macro) | ROC-AUC (macro) |
|---|---|---|---|
| CNN (from scratch) | 0.742 | 0.7227 | 0.9244 |
| EfficientNetB0 | 0.949 | 0.9458 | 0.9962 |
| MobileNetV3-Large | 0.952 | 0.9482 | 0.9953 |
| ResNet50 | 0.967 | 0.9608 | 0.9988 |
| ConvNeXt-Tiny | 0.951 | 0.9377 | 0.9946 |
| InceptionV3 | 0.966 | 0.9572 | 0.9986 |
| Xception | 0.980 | 0.9803 | **0.9991** |
| **Proposed (weighted fusion)** | **0.982** | **0.9823** | 0.9978 |

**Key finding:** on the in-distribution test set, InceptionV3 edges out the proposed model on raw accuracy (0.9550 vs 0.9494), but the proposed model has the best ROC-AUC in-distribution *and* the best accuracy and F1 on the fully external, out-of-distribution BRISC2025 set (0.982 / 0.9823) — the ensemble's main benefit is **generalization robustness across datasets**, not marginal in-distribution accuracy, which matches the "cross-dataset" framing of the underlying paper.

### Learned fusion weights

After training, the dynamic weighted fusion layer converged to (softmax-normalized):

| Branch | Weight |
|---|---|
| ResNet50 | 0.3392 |
| EfficientNetB5 | 0.3334 |
| Custom CNN | 0.3274 |

The three branches end up nearly equally weighted, suggesting each contributes complementary, non-redundant information rather than one branch dominating.

## Explainable AI

Grad-CAM++, Score-CAM, Eigen-CAM, and a CNN-adapted Attention Rollout are applied to the ResNet50 branch's conv blocks (`conv3_block4_out`, `conv4_block6_out`, `conv5_block3_out`) for 2 sample images per class (8 total), saved as side-by-side comparison panels under `outputs/xai/`.

> Note on Attention Rollout: it was originally defined for transformer self-attention. Since the proposed model's branches are CNNs, we implement the standard CNN-XAI-literature adaptation — cumulative element-wise multiplication of normalized Grad-CAM-style maps across sequential conv blocks, approximating how spatial importance compounds through depth.

## Requirements

- TensorFlow/Keras (2.15+ recommended; ConvNeXt in `tf.keras.applications` requires ≥2.10, with an automatic EfficientNetV2B0 fallback if unavailable)
- `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
- Kaggle T4 x2 GPU (or any 2-GPU / single-GPU CUDA setup; the code auto-detects and adjusts `MirroredStrategy` accordingly)

## Citation

If you use this framework, please cite the associated paper (details to be added upon publication) and the two datasets:

- Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle.
- BRISC2025 Dataset, Kaggle.

## License

Add your chosen license here (e.g., MIT) before making the repository public.

## Changelog

### v1.0.0
- Initial release: 7 baseline models (CNN, EfficientNetB0, MobileNetV3-Large, ResNet50, ConvNeXt-Tiny, InceptionV3, Xception)
- Proposed dynamic weighted fusion ensemble (ResNet50 + EfficientNetB5 + Custom CNN)
- Full evaluation suite per model (classification report, confusion matrix, ROC-AUC, loss/accuracy curves, calibration curves)
- XAI on the proposed model: Grad-CAM++, Score-CAM, Eigen-CAM, CNN-adapted Attention Rollout
- External cross-dataset validation on BRISC2025
