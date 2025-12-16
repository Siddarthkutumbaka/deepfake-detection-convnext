# Experiments Log — Deepfake Detection (ConvNeXt)

This document tracks all experiments conducted for reproducibility,
ablation studies, and future publication.

---

## Dataset
- DFDC (DeepFake Detection Challenge)
- Modality: Face frames extracted from videos
- Labels: Real / Fake

---

## Baselines
- CNN (simple)
- ResNet-18
- EfficientNet-B0

---

## Main Model
- ConvNeXt-Tiny
- Pretrained: ImageNet
- Input size: 224x224

---

## Experiments Plan
- [ ] Baseline vs ConvNeXt comparison
- [ ] Frames per video ablation (5 / 10 / 20)
- [ ] Input resolution ablation (224 vs 299)
- [ ] Threshold tuning (ROC-based)
- [ ] Cross-video generalization

---

## Metrics
- Accuracy
- ROC-AUC
- Precision / Recall
- Confusion Matrix

---

## Notes
(To be filled after experiments)
