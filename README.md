# Forest Sentinel AI — Comprehensive Forest Fire Intelligence System

> An end-to-end, AI-powered forest fire detection and risk prediction platform. Built utilizing PyTorch for deep learning (Computer Vision) and Scikit-Learn/XGBoost for tabular environmental data analysis, presented via a professional Streamlit interface.

---

## 1. Project Overview

Forest fires cause catastrophic ecological and economic damage. Early detection and risk prediction can dramatically reduce fire spread and loss of life. This system approaches the problem from two angles:
1. **Visual Detection**: Real-time identification of active fires from aerial drone imagery.
2. **Environmental Risk Profiling**: Predictive analysis of fire risk and estimated burn area using meteorological sensors (FWI indices).

---

## 2. Dataset Architecture & Engineering

### 2.1 Image Dataset Pipeline
We utilized two primary image datasets:
1. **Kaggle `fire_dataset`**: Clean, high-quality images.
2. **UAVS-FDDB Raw Images**: Real-world aerial drone footage of forest fires.

**Dataset Engineering Steps:**
* **Leakage Prevention**: We deliberately excluded the `UAVS-FDDB Augmented` folder (15,560 images) because including pre-augmented images introduces massive data leakage between train and test sets.
* **Deduplication**: We generated MD5 hashes for every image and aggressively removed **2,566 exact duplicates**.
* **Final Pool**: Left with exactly **2,492 unique images**.
* **Data Split**: Stratified 60% Train (1,494), 20% Validation (499), 20% Test (499).
* **Imbalance Handling**: The dataset favored `FIRE` (3:1 ratio). We calculated dynamic class weights: `FIRE = 0.669`, `NO_FIRE = 1.981`.

### 2.2 Tabular Dataset (`forestfires.csv`)
Sourced from the UCI Machine Learning Repository (Cortez and Morais, 2007).
* **Dimensions**: 518 rows, 13 features.
* **Features Used**: Spatial (X, Y), Temporal (month, day), FWI Indices (FFMC, DMC, DC, ISI), Meteorological (temp, RH, wind, rain).
* **Target 1 (Binary)**: `fire_occurred` (derived from `area > 0`). 
* **Target 2 (Regression)**: `area` (burned hectares).
* **Engineering**: 
  - Log-transformation (`log1p`) of the `area` target to mitigate severe right-skew/zero-inflation (47.8% of records are 0.0).
  - Label Encoding of categorical features (`month`, `day`).
  - Standard Scaling of numerical features.

---

## 3. Computer Vision Model (Image Classification)

### 3.1 Model Benchmarking
We benchmarked four CNN architectures on a rapid 5-epoch test.
| Model | Accuracy | Precision | Recall | F1-Score | Inference (ms) |
|-------|----------|-----------|--------|----------|----------------|
| **EfficientNet-B0** | 98.50% | 0.9851 | 0.9850 | **0.9850** | 13.38 ms |
| ResNet18 | 98.50% | 0.9850 | 0.9850 | 0.9850 | 17.88 ms |
| MobileNetV2 | 98.50% | 0.9851 | 0.9850 | 0.9850 | 10.03 ms |
| MobileNetV3 | 97.50% | 0.9752 | 0.9750 | 0.9751 | 3.43 ms |

**Selection**: `EfficientNet-B0` was selected for its superior F1 convergence and high parameter efficiency.

### 3.2 Final Training Configurations & Hyperparameters
* **Architecture**: EfficientNet-B0 (pretrained on ImageNet).
* **Fine-Tuning Strategy**: Feature extractor blocks frozen; last two feature blocks and the fully-connected classifier layers unfrozen for domain adaptation.
* **Image Dimensions**: $224 \times 224$ pixels.
* **Batch Size**: 16.
* **Optimizer**: AdamW.
* **Learning Rate**: $1 \times 10^{-3}$ with Weight Decay of $1 \times 10^{-4}$.
* **Scheduler**: CosineAnnealingLR (T_max=20, eta_min=1e-6).
* **Loss Function**: CrossEntropyLoss utilizing the predefined class weights (0.669 / 1.981).
* **Epochs**: Maximum 20, equipped with Early Stopping (Patience = 5 epochs monitoring Validation F1).
* **Augmentations (Train Only)**: RandomCrop(224), RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.2), RandomRotation(10°), ColorJitter(brightness/contrast=0.2), RandomGrayscale(p=0.02).
* **Normalization**: standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

### 3.3 Final Image Model Test Performance
* **Best Epoch**: Epoch 8 (Early stopping triggered).
* **Test Accuracy**: **99.00%**
* **Test F1-Score**: **0.9900**
* **Test ROC-AUC**: **0.9919**
* **Inference Speed**: ~23.72ms per image (CPU).
* **Confusion Matrix**: 369 True Positives, 125 True Negatives, 4 False Positives, 1 False Negative.

---

## 4. Tabular Models (Environmental Profiling)

### 4.1 Tabular Benchmarking
Tested models included XGBoost, Random Forest, HistGradientBoosting, and Logistic Regression via 5-Fold Stratified Cross Validation.

**Classifier Selection**: Random Forest Classifier.
**Regressor Selection**: Random Forest Regressor.

### 4.2 Final Tabular Test Performance
* **Classifier Accuracy**: 62.50%
* **Classifier Precision**: 63.64%
* **Classifier Recall**: 64.81%
* **Classifier F1-Score**: 0.6422
* **Classifier ROC-AUC**: 0.6978
* **Regressor MAE**: 12.27 ha (Target `area` mapped back from log-space via `expm1`).

> Note: The tabular results mirror the global baseline for the UCI `forestfires.csv` dataset, which is historically difficult to predict accurately due to heavy zero-inflation and stochastic variance. 

---

## 5. Explainable AI (XAI) Integration

To ensure the AI decisions are interpretable by forest rangers and operators, we implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**. 
During image inference, the dashboard hooks into the final convolutional layer of the EfficientNet-B0 backbone. It calculates the gradients of the target class (`FIRE`) flowing into this layer to produce a localization heatmap. This heatmap is overlaid onto the original image to visually highlight the specific flames/smoke that triggered the model.

---

## 6. The Streamlit Dashboard (`app.py`)

A state-of-the-art, hardware-resilient web application with an elegant **White / Light Theme**:
* **Aesthetic**: Modern pure white card layout on slate-50 canvas with emerald (`#059669`) and crimson (`#dc2626`) risk accents.
* **Module 1: Vision Fire Detection**: Upload aerial/drone images or pick 1-click built-in test samples. Outputs instant classification with confidence spectrum, Grad-CAM attention heatmap overlay with customizable colormaps (`jet`, `inferno`, `viridis`, etc.) and alpha transparency, plus an operational response playbook.
* **Module 2: Live Weather Sandbox**: Real-time interactive simulator with sliders for FWI indices (`FFMC`, `ISI`, `DMC`, `DC`) and weather variables (`temp`, `RH`, `wind`, `rain`) with 4 preset meteorological scenarios (*Scorching Heatwave*, *Dry Spring Winds*, etc.) for instant risk prediction and burned area estimation.
* **Module 3: Batch CSV Risk Analysis**: Bulk analyze environmental data files or run the built-in 20-sample demo suite with donut risk charts, styled data tables, and CSV export.
* **Module 4: Analytics & Benchmarks**: High-resolution light-theme visualizations of dataset distributions, CNN benchmark comparisons (EfficientNet-B0, MobileNetV3, ResNet18), loss/accuracy curves, and XGBoost feature importance rankings.
* **Module 5: Dataset Inventory & Architecture**: Interactive asset explorer for Archive, UAVS-FDDB, and UCI tabular corpora.

---

## 7. Execution & Launch Instructions

**1. Setup Environment**
```bash
# Ensure Python 3.9+ is installed
pip install -r requirements.txt
```

**2. Launch the Dashboard**
```bash
cd "Implementation"
streamlit run dashboard/app.py
```

**3. Model Retraining (Optional)**
If you wish to retrain the models from scratch, execute the provided Jupyter notebooks in exact sequential order:
```
1. 01_dataset_audit.ipynb
2. 02_image_data_preprocessing.ipynb
3. 03_image_model_benchmarking.ipynb
4. 04_final_image_model_training.ipynb (or train_image_model.py)
5. 05_image_model_evaluation_explainability.ipynb
6. 06_tabular_data_analysis.ipynb
7. 07_tabular_model_benchmarking.ipynb
8. 08_final_tabular_model_training.ipynb
9. 09_tabular_model_evaluation.ipynb
10. 10_final_inference_testing.ipynb
```

---
*Developed as a comprehensive AI implementation task for Forest Fire intelligence and detection.*
