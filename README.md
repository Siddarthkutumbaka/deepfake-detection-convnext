# Deepfake Detection using ConvNeXt

This project explores **deepfake video detection** using a modern ConvNeXt-based image classifier trained on the **DFDC (DeepFake Detection Challenge)** dataset.  
The goal is to evaluate whether strong image backbones can effectively detect manipulated facial frames without explicit temporal modeling.

---

## 📌 Motivation
Deepfake videos pose serious threats to:
- Digital media authenticity
- Political misinformation
- Identity fraud

This project investigates:
- Frame-level deepfake detection
- Majority voting across frames
- Model confidence and calibration

---

## 🧠 Methodology
- Backbone: **ConvNeXt-Tiny (ImageNet pretrained)**
- Input: Sampled face frames from videos
- Output: Binary classification (REAL vs FAKE)
- Aggregation: Majority voting across frames
- Loss: Cross-entropy
- Metrics: Accuracy, ROC-AUC, confusion matrix

---

## 📂 Dataset
- **DFDC (DeepFake Detection Challenge)**
- Real and manipulated face frames
- Dataset accessed via Kaggle

> Dataset is **not included** in this repository.

---

## 📊 Results (Validation)
| Metric | Value |
|------|------|
| Validation Accuracy | ~84% |
| ROC-AUC | ~0.81 |
| Mean FAKE Prob (REAL frames) | ~0.68 |
| Mean FAKE Prob (FAKE frames) | ~0.86 |

---

## 🔬 Observations
- ConvNeXt learns strong spatial artifacts
- Frame-level predictions benefit from aggregation
- Some REAL videos show high FAKE confidence → motivates calibration & temporal modeling

---

## 📁 Repository Structure

deepfake-detection-convnext/
│
├── notebooks/        # Kaggle notebooks (training & analysis)
├── src/              # Clean training & inference code
├── results/          # Plots, metrics, confusion matrices
├── requirements.txt
├── .gitignore
└── README.md

---

## 🚀 Future Work
- Temporal modeling (CNN + LSTM / Transformers)
- Face alignment & quality filtering
- Cross-dataset generalization
- Model calibration & uncertainty estimation

---

## 👤 Author
**Harshith Siddartha Kutumbaka**  
MS Computer Science (Data Science)  
UNC Charlotte
