# Explainable Cost-Sensitive Gradient Tree Boosting and Dual-Tier TreeSHAP for Non-Invasive Gallbladder Disease Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Review--II-100%25%20Verified-green.svg)]()

Final-Year B.Tech Capstone Project in Computer Science and Engineering  
**Institution:** Vellore Institute of Technology (VIT), Chennai, Tamil Nadu, India  

---

## 👥 Authors
* **Kantipudi Venkat Rohith**
* **Gajjala Karthik Reddy**
* **Harsha Lingampalli**
* **Tarun Tej**

---

## 📌 Project Overview
This repository contains the official dataset, preprocessing pipelines, model architectures, and TreeSHAP explainability modules for non-invasive gallbladder disease screening using routine peripheral blood biochemistry.

### Key Highlights:
1. **CDC NHANES Cohort:** Analyzes 6,629 adult participants ($5,894$ controls, $735$ disease cases; 1:8.02 class imbalance).
2. **Physiological Feature Engineering:** Implements direct-to-total bilirubin ratio ($R_{\text{bil}}$), De Ritis ratio ($R_{\text{DeRitis}}$), and non-HDL cholesterol.
3. **Cost-Sensitive Learning:** Inverse class weighting ($w_0 = 0.562, w_1 = 4.509$; penalty ratio $8.02\times$) embedded directly into loss gradients.
4. **Holdout Performance:** Achieves **96.61% accuracy, 91.16% sensitivity, 97.29% specificity, and 0.989 ROC-AUC** on an independent test partition ($N_{\text{test}} = 1,326$).
5. **Dual-Tier TreeSHAP:** Provides global population beeswarm importance and localized patient-level waterfall diagnostic attributions.

---

## 📂 Repository Structure
```
├── data/
│   └── nhanes_gallbladder_cohort.csv      # Continuous CDC NHANES cohort (N=6,629)
├── figures/
│   ├── architecture_diagram.png           # 5-Layer clinical AI architecture
│   ├── class_distribution.png             # Target imbalance distribution
│   ├── correlation_heatmap.png            # Biomarker Pearson correlation
│   ├── confusion_matrix.png               # Holdout confusion matrix
│   ├── model_comparison.png               # Benchmark comparisons
│   ├── roc_curves.png                     # Multi-model ROC curves
│   ├── precision_recall.png               # PR curves
│   ├── shap_summary.png                   # Global TreeSHAP summary beeswarm
│   ├── shap_global_importance.png         # Mean absolute SHAP ranking
│   └── shap_waterfall.png                 # Patient-level waterfall attribution
├── src/
│   ├── data_preprocessing.py              # Cohort cleaning and Z-score scaling
│   ├── feature_engineering.py             # Ratio derivation formulations
│   ├── model_training.py                  # Cost-sensitive XGBoost training
│   ├── evaluate_models.py                 # Multi-metric holdout evaluation
│   └── treeshap_explainability.py         # Exact TreeSHAP game-theoretic explainers
├── main.py                                # End-to-end execution pipeline
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Rohith-stack-creater/Explainable-Clinical-AI-Gallbladder-Prediction.git
cd Explainable-Clinical-AI-Gallbladder-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python main.py
```
