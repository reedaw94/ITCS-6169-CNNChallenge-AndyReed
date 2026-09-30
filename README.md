# ITCS 6169/8169 Assignment 1 – CNN Scene Classification

## Overview

This project was completed for ITCS 6169/8169 and investigates convolutional neural networks (CNNs) for scene classification.

The task is to classify images into 16 scene categories using a dataset containing 2,400 training images and 400 test images. The project explores the effects of RGB input, data augmentation, increased CNN depth, longer training, and learning-rate scheduling.

The final model is a custom CNN trained from scratch using PyTorch.

## Scene Categories

The dataset contains the following 16 classes:

- Bedroom
- Coast
- Flower
- Forest
- Highway
- Industrial
- InsideCity
- Kitchen
- LivingRoom
- Mountain
- Office
- OpenCountry
- Store
- Street
- Suburb
- TallBuilding

## Experimental Results

Four configurations were evaluated during development.

| Experiment | Main Change | Best Validation Accuracy |
|---|---|---:|
| Baseline | Small TNet with grayscale input | 49.79% |
| Experiment 1 | RGB input and data augmentation | 53.96% |
| Experiment 2 | Deeper custom CNN, 20 epochs | 51.46% |
| Experiment 3 | Deeper CNN, 40 epochs, LR scheduler | **64.79%** |

Experiment 3 produced the best validation performance and was selected as the final model.

## Final Model Architecture

The final model is a custom convolutional neural network operating on 64 × 64 RGB images.

The feature extractor contains:

- Two 3 × 3 convolutional layers with 32 channels
- 2 × 2 max pooling
- Two 3 × 3 convolutional layers with 64 channels
- 2 × 2 max pooling
- One 3 × 3 convolutional layer with 128 channels
- 2 × 2 max pooling
- Adaptive average pooling

The classifier contains:

- Flatten layer
- Dropout with `p = 0.30`
- Fully connected layer with 16 outputs

The final model contains approximately **141,488 trainable parameters**.

## Training Configuration

The final Experiment 3 configuration used:

| Parameter | Value |
|---|---|
| Image size | 64 × 64 RGB |
| Batch size | 64 |
| Epochs | 40 |
| Optimizer | Adam |
| Initial learning rate | 0.002 |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 3 |
| Dropout | 0.30 |
| Random seed | 0 |

The learning-rate scheduler monitored validation loss and reduced the learning rate when validation loss stopped improving.

The best model checkpoint was selected according to validation accuracy rather than automatically using the final training epoch. The highest validation accuracy was **64.79% at epoch 39**.

## Data Preprocessing and Augmentation

All images are resized to 64 × 64 pixels and converted to PyTorch tensors.

Training images use:

- RGB input
- Random horizontal flip with probability 0.5
- Random rotation of ±10 degrees
- Normalization with mean `[0.5, 0.5, 0.5]`
- Normalization with standard deviation `[0.5, 0.5, 0.5]`

Validation and test images use deterministic preprocessing:

- RGB input
- Resize to 64 × 64
- Tensor conversion
- Normalization

Random augmentation is not applied during validation or testing.

## Train/Validation Split

The 2,400 provided training images are divided into:

- **1,920 training images (80%)**
- **480 validation images (20%)**

A fixed random seed of `0` is used so that the same train/validation split can be reproduced.

The provided 400-image test set is kept separate from the training and validation data.

## Repository Structure

The repository is organized as follows:

```text
ITCS6169-Assignment1/
│
├── README.md
├── AI_USAGE.md
├── requirements.txt
├── assignment1.py
├── evaluate.py
│
├── checkpoints/
│   └── deepercnn_exp3_best.pth
│
└── results/
    ├── exp3_loss.png
    ├── exp3_validation_accuracy.png
    └── exp3_learning_rate.png
