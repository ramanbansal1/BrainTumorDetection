# BrainTumorDetection
This project implements a brain tumor detection system using the VGG-16 convolutional neural network (CNN) architecture. The model leverages TensorFlow for deep learning and employs data augmentation techniques to improve model generalization and accuracy.

---
## Table of Contents

1. Overview
2. Prerequisites
3. Dataset
4. Model Architecture
5. Data Augmentation
6. Training and Evaluation
7. Results


---

## Overview

This project aims to classify brain MRI images as having a tumor or not. The VGG-16 architecture is fine-tuned to extract features and make predictions on MRI scan images. Data augmentation is applied to enhance the training process and reduce overfitting.

---

## Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- imutils
---

## Dataset

You can download the dataset from Kaggle using the following URL:

**Dataset URL:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

Ensure the dataset is structured as follows:

```
/dataset
  /yes
  /no
```
---

## Model Architecture

The model uses the VGG-16 architecture pre-trained on ImageNet as the base model. The following modifications are made:

- The fully connected layers are replaced with a custom classification head.
- Dropout is added to reduce overfitting.
- Softmax activation is used for binary classification.

---

## Data Augmentation

To improve the model's robustness, data augmentation is applied using the following transformations:

- Random rotations
- Horizontal flipping
- Changing Brightness
- Shifting
- rotation
- shearing

These transformations are implemented using TensorFlow's `ImageDataGenerator`.

---

## Training and Evaluation

1. The data is split into 70% for training and half of 30% for testing and rest for validation.
2. Use data augmentation is used for reducing overfitting
3. Evaluate the model's performance using accuracy, precision, recall, and F1-score, AUC score.
---

## Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 87%   |
| Precision | 82%   |
| Recall    | 95%   |
| F1-Score  | 0.88  |
| AUC-Score | 0.91  |

---

