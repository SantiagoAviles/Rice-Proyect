# Deep Learning Models for Defect Identification in Oryza Sativa Rice Grains

> **Work:** *Deep Learning Models for Defect Identification in Oryza Sativa Rice Grains: A Comparative Study* — submitted to *AgriEngineering* (2026).
> Yasiel Pérez Vera, Melissa Kristel Chambi Flores, Santiago Alonso Avilés Córdova, Irvin Estuardo Cazorla Macedo, Percy Aaron Luján Biamonte, Edgardo Alfredo Rivero Callohuanca.
> Universidad Católica de Santa María, Arequipa, Perú.

This repository contains the full experimental pipeline for the automatic classification of visual defects in Peruvian (*Oryza sativa*) rice grains using transfer learning and convolutional neural networks (CNNs). Five pretrained architectures are systematically compared under identical conditions, addressing a gap in the literature on multiclass defect classification of Peruvian rice with real-world RGB imagery.

---

## Overview

Manual grain inspection is slow, inconsistent, and hard to scale. This project evaluates whether CNN-based transfer learning can reliably distinguish four grain quality categories from standard RGB photos, using a dataset collected at a mill in Lambayeque, Peru.

**Key results on the independent test set:**

| Model | Accuracy | Loss | 95% CI |
|---|---|---|---|
| **ResNet50** | **84.71%** | **0.3765** | [82.47%, 86.95%] |
| EfficientNetB0 | 83.60% | 0.3845 | [81.30%, 85.90%] |
| DenseNet121 | 83.20% | 0.3996 | [80.87%, 85.52%] |
| MobileNetV2 | 82.09% | 0.4584 | [79.71%, 84.48%] |
| InceptionV3 | 80.89% | 0.4535 | [78.44%, 83.33%] |

Pairwise McNemar tests (α = 0.05) showed no statistically significant difference between ResNet50, EfficientNetB0, and DenseNet121 — confirming that multiple architectures are competitive under the same transfer learning framework.

---

## Dataset

**Source:** [Peruvian Rice Dataset — Kaggle](https://www.kaggle.com/datasets/cristhiansempertegui/dataset-de-arroz-peruano)

- 6,599 PNG images at 400 × 399 px, each showing a single grain on a uniform background.
- Collected at a mill in Lambayeque, Peru, for quality classification purposes.
- A cleaning phase removed corrupted files before any processing.

### Classes

| Class | Description | Original count |
|---|---|---|
| **Whole** | Intact grain, no visible defects | 3,352 |
| **Stained** | Dark discoloration, fungal or chemical origin | 857 |
| **Broken** | Clear structural fracture | 867 |
| **Chalky** | Opaque / chalk-like endosperm | 1,523 |

The dataset is heavily imbalanced (Whole class represents ~51% of samples), which motivates the augmentation and class-weighting strategy described below.

---

## Methodology

The pipeline follows a strict sequence to prevent data leakage: **split first, then augment**.

```
Original Dataset (6,599 images)
        │
        ▼
   Data Cleaning
        │
        ▼
Stratified Split 70 / 15 / 15
   ┌────┴─────┬──────────────┐
Training    Validation     Test
  (70%)      (15%)         (15%)
   │        kept original  kept original
   ▼
Data Augmentation (train only)
   │  rotation, shift, shear, zoom, flip
   ▼
Balanced Training Set (9,007 images)
   │
   ▼
Two-Phase Transfer Learning ──────────────────────┐
   Phase 1: frozen backbone, train head only       │
   Phase 2: unfreeze last 10 layers, fine-tune     │
                                                   ▼
                                        Comparative Evaluation
                                        on independent test set
```

### Dataset split after augmentation

| Class | Training (aug.) | Validation (original) | Test (original) |
|---|---|---|---|
| Whole | 2,346 | 502 | 504 |
| Stained | 2,184 | 128 | 130 |
| Broken | 2,208 | 130 | 131 |
| Chalky | 2,269 | 228 | 229 |
| **Total** | **9,007** | **988** | **994** |

Validation and test sets are never touched by augmentation — they preserve the real-world class distribution.

### Data augmentation (training set only)

Applied with Keras `ImageDataGenerator`:

- Random rotation (±25°)
- Width / height shift (±15%)
- Shear range (0.15)
- Zoom range
- Horizontal and vertical flipping

### Classification head (shared across all models)

```
GlobalAveragePooling2D
→ Dense(256, relu)
→ BatchNormalization
→ Dropout(0.4)
→ Dense(128, relu)
→ Dropout(0.3)
→ Dense(4, softmax)
```

### Training hyperparameters

| Parameter | Phase 1 (head only) | Phase 2 (fine-tuning) |
|---|---|---|
| Frozen layers | All backbone layers | All except last 10 |
| Learning rate | 1 × 10⁻⁴ | 1 × 10⁻⁵ |
| Optimizer | Adam | Adam |
| Loss | Categorical cross-entropy | Categorical cross-entropy |
| Batch size | 32 | 32 |
| Max epochs | 20 | 20 |
| Early stopping patience | 3 (val_loss) | 5 (val_loss) |
| ReduceLROnPlateau | factor 0.3, patience 2, min_lr 1e-6 | factor 0.5, patience 3, min_lr 1e-7 |
| Class weighting | Balanced (inversely proportional to frequency) | — |
| Input size | 224 × 224 | 224 × 224 |

The same unfreezing strategy (last 10 layers) is applied uniformly to all five backbones to ensure a fair comparison regardless of total network depth.

### Evaluated architectures

| Architecture | Design paradigm |
|---|---|
| MobileNetV2 | Inverted bottlenecks + depthwise-separable convolutions for edge deployment |
| EfficientNetB0 | Compound scaling of depth, width, and resolution |
| ResNet50 | Residual (skip) connections to solve vanishing gradients |
| DenseNet121 | Dense connectivity — each layer receives feature maps from all previous layers |
| InceptionV3 | Parallel convolutions at multiple scales via Inception modules |

All backbones are initialized with ImageNet weights (transfer learning).

---

## Results

### Per-class F1-score on the test set

| Model | Whole | Stained | Broken | Chalky |
|---|---|---|---|---|
| ResNet50 | 0.8643 | **0.9354** | 0.8872 | **0.7352** |
| EfficientNetB0 | 0.8511 | 0.9278 | **0.9004** | 0.7108 |
| DenseNet121 | 0.8492 | 0.9308 | 0.8906 | 0.7056 |
| MobileNetV2 | 0.8433 | 0.8750 | 0.8945 | 0.6987 |
| InceptionV3 | 0.8350 | 0.9070 | 0.8945 | 0.6362 |

Stained and Broken grains are classified most reliably. **Chalky grains are the hardest class** across all models — chalky endosperm shares the external morphology of intact (Whole) grains, so the discriminating signal reduces to subtle differences in reflectance and opacity that RGB images partially capture.

### Pairwise McNemar tests (α = 0.05)

| Model A | Model B | p-value | Significant |
|---|---|---|---|
| MobileNetV2 | EfficientNetB0 | 0.2513 | No |
| MobileNetV2 | ResNet50 | 0.0279 | **Yes** |
| MobileNetV2 | DenseNet121 | 0.3930 | No |
| MobileNetV2 | InceptionV3 | 0.3904 | No |
| EfficientNetB0 | ResNet50 | 0.3382 | No |
| EfficientNetB0 | DenseNet121 | 0.7843 | No |
| EfficientNetB0 | InceptionV3 | 0.0249 | **Yes** |
| ResNet50 | DenseNet121 | 0.1591 | No |
| ResNet50 | InceptionV3 | 0.0027 | **Yes** |
| DenseNet121 | InceptionV3 | 0.0692 | No |

---

## Repository Structure

```
.
├── RiceModelsOfficial.ipynb    # Full pipeline: download → split → augment
│                               # → train (all 5 models) → evaluate → compare
├── rice_dataset.rar            # Pointer to original Kaggle dataset
├── rice_dataset_augmented.rar  # Pointer to augmented training set
├── .gitattributes              # Git LFS configuration
├── LICENSE                     # MIT License
└── README.md                   # This file
```

The notebook `RiceModelsOfficial.ipynb` is self-contained and runs on Google Colab. It covers:

1. Kaggle dataset download
2. Image consolidation and class mapping
3. Stratified 70 / 15 / 15 split
4. Offline data augmentation (training set only)
5. Image integrity verification (corrupt file removal)
6. Two-phase transfer learning for all five architectures
7. Confusion matrices, per-class metrics, and learning curves
8. 95% confidence intervals (Wilson score)
9. Pairwise McNemar statistical tests

---

## Reproducing the Results

1. Open `RiceModelsOfficial.ipynb` in Google Colab.
2. Enable a GPU runtime (T4 or higher recommended).
3. Upload your `kaggle.json` API token when prompted (Kaggle → Settings → Create New API Token).
4. Run all cells in order — the notebook downloads the dataset, builds the splits, augments, trains, and evaluates automatically.

No manual path changes are needed when running on Colab with the default `/content/` structure.

---

## Dependencies

| Library | Version used |
|---|---|
| Python | 3.12.13 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| Scikit-learn | 1.6.1 |
| Pandas | 2.2.2 |
| NumPy | 2.0.2 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Pillow | 11.3.0 |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Funded by the **Universidad Católica de Santa María**, Arequipa, Perú. We thank the Professional School of Systems Engineering for their support.
