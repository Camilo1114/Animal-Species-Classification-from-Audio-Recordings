# Animal Species Classification from Audio Recordings

## Overview

This project compares two deep learning approaches for bird species classification using audio recordings from the BirdCLEF 2025 dataset:

- Deep Variational Autoencoder (DVAE) with a classification head  
- Convolutional Neural Network (CNN) trained directly on MFCC features  

The objective is to evaluate the trade-offs between generative and discriminative learning for ecoacoustic monitoring.

---

## Dataset

- Source: BirdCLEF 2025 (Kaggle)  
- Location: El Silencio Natural Reserve, Colombia  
- Subset: 15 bird species  
- Audio converted from `.ogg` to `.wav`  
- Resampled to 16 kHz  

---

## Feature Extraction

- 13 MFCC coefficients  
- 40 mel filters  
- Sequences padded or trimmed to fixed length  

MFCC tensors were used as model inputs.

---

## Models

### DVAE

- GRU encoder  
- 64-dimensional latent space  
- GRU decoder  
- MLP classification head  

**Loss components**
- Reconstruction loss (MSE)  
- KL divergence  
- Cross-entropy loss  

**Training**
- 1000 epochs  
- Adam optimizer (lr = 1e-3)  
- KL annealing during initial epochs  

---

### CNN

- 3 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)  
- Fully connected layers  
- Softmax output (15 classes)  

**Training**
- 100 epochs  
- Adam optimizer (lr = 1e-4)  
- Step learning rate scheduler  
- Cross-entropy loss  

**Data augmentation**
- Gaussian noise  
- Time masking  
- Frequency masking  

---

## Results

| Model        | Test Accuracy |
|--------------|--------------|
| CNN          | 68.77%      |
| DVAE + MLP   | 50.21%      |

The CNN achieved higher classification accuracy. The DVAE learned structured latent representations but showed weaker class separation.
