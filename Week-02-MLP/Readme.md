# Experiment 2 — Multi-Layer Perceptron for Multi-Class Image Classification

**Course:** CS3807 – Deep Learning Laboratory
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science
**Name:** Srivatsan S
**Register No.:** 24011101093
**Class:** AIDS – B


## Overview

This repository implements a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for multi-class image classification on the **Fashion-MNIST** dataset, along with automated hyperparameter optimization and a supplementary study on the linear-separability limitations of single-layer perceptrons using the XOR problem.


## What's Implemented

### 1. Fashion-MNIST MLP Classification
- Dataset exploration (60,000 train / 10,000 test images, 10 classes)
- Preprocessing: flattening (28×28 → 784), pixel normalization, one-hot encoding
- Baseline architecture: `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
- Training for 20 epochs (Adam optimizer, batch size 32)
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, classification report

### 2. Automated Hyperparameter Optimization
- `RandomizedSearchCV` (5-fold cross-validation) via the SciKeras wrapper
- Search space: hidden layers, hidden neurons, learning rate, batch size, epochs, optimizer, activation function, dropout rate
- Best model retrained on the full training set and compared against the baseline

### 3. XOR — SLP vs. MLP
- Single-Layer Perceptron trained on XOR to demonstrate **non-convergence** due to linear inseparability
- Decision boundary plotted after each weight update
- MLP (2 → 2 hidden → 1, sigmoid, backpropagation from scratch) trained on the same problem to show convergence
- Written analysis of why a straight-line decision boundary can never separate XOR's diagonally opposite classes

## Results Summary

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 88.22% | 88.51% |
| Precision | 88.28% | 88.67% |
| Recall | 88.22% | 88.51% |
| F1-score | 88.17% | 88.50% |

**Best hyperparameters:** 1 hidden layer, 256 neurons, ReLU activation, Adam optimizer, learning rate 0.001, batch size 16, dropout 0.2
