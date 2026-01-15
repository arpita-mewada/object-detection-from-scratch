# Object Detection from Scratch (PASCAL VOC Subset)

## Overview
This project demonstrates a complete end-to-end object detection pipeline built **from scratch** using PyTorch.
The goal is **not SOTA accuracy**, but to validate **engineering correctness** across data handling, training,
evaluation, and inference.

> **The goal is not SOTA accuracy but validation of a complete detection pipeline trained from scratch.**

---

## Dataset
- Dataset: PASCAL VOC 2007 (subset)
- Classes: person, car, dog
- Train: ~300 images
- Validation: ~100 images
- Test: ~100 images

*A reduced subset was used to lower training cost while demonstrating the full pipeline.*

---

## Model Architecture
Custom single-stage CNN detector (from scratch):

- Input: 224×224×3
- Backbone:
  - Conv(16) + ReLU + MaxPool
  - Conv(32) + ReLU + MaxPool
  - Conv(64) + ReLU
- Detection Head:
  - Bounding box regression (x, y, w, h)
  - Classification (3 classes)
- Adaptive pooling used for stable feature dimensions

No pretrained weights were used.

---

## Training Setup
| Item | Value |
|----|----|
| Framework | PyTorch |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 8 |
| BBox Loss | Smooth L1 |
| Class Loss | Cross Entropy |
| Input Size | 224×224 |

---

## Evaluation Metrics
- **FPS (CPU):** ~349.49 FPS  
- **Model Size:** ~3.16 MB  
- **Metric Focus:** mAP@0.5 (simplified), FPS, model size

---

## Inference
- Inference performed on unseen images
- Bounding boxes and class labels drawn using OpenCV
- Results saved to `demo/output/`

---

## Results
- 5–10 inference images generated
- Demonstrates full deployment pipeline

---

## Notes
- This project focuses on pipeline correctness rather than accuracy
- Designed to be compatible with x86_64 and ARM CPUs
