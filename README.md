# voice-emotion-recognition

This repository contains an **end-to-end Voice Emotion Recognition system**, covering data processing, feature engineering, model training, evaluation, and cloud deployment using **Azure Machine Learning**.

The project compares **classical machine learning approaches** with state-of-the-art **deep learning (Wav2Vec2)** and culminates in a **real-time Azure ML online endpoint** for inference.

## Overview
The goal of this project is to predict human emotions from speech audio using two complementary approaches:

1. **Classical ML Pipeline**
   - Extracts and aggregates audio features (MFCCs, Chroma STFT, Spectral Contrast, Tonnetz, Zero-Crossing Rate, RMS, etc.)
   - Converts features into tabular CSV datasets
   - Trains multiple classical models (Random Forest, SVM, Logistic Regression) using Microsoft Azure AutoML
   - Provides a cost-efficient CPU-only baseline

2. **Deep Learning Pipeline**
   - Uses **raw audio files** directly
   - Fine-tunes a pretrained **Wav2Vec2** model for emotion classification
   - Achieves higher accuracy (~80–82%) by learning complex audio patterns
   - Requires GPU for training

It demonstrates trade-offs between cost, complexity, and performance, and showcases a production-style ML deployment pipeline on Azure.   

## Classical Machine Learning Pipeline
### Feature Engineering
Extracted audio features include:
- MFCCs
- Chroma STFT
- Spectral Contrast
- Tonnetz
- Zero-Crossing Rate
- RMS Energy
Features are aggregated into tabular CSV datasets for model training.

### Models Trained (Azure AutoML)
- Random Forest
- Support Vector Machine (RBF kernel)
- Logistic Regression

All models were trained using Azure ML AutoML (classification) on CPU compute, providing a cost-efficient baseline.

## Deep Learning Pipeline (Wav2Vec2)
- Uses raw audio waveforms directly
- Fine-tunes a pretrained Wav2Vec2 base model
- Learns complex temporal and spectral patterns beyond handcrafted features
- Trained using PyTorch + Hugging Face Transformers
- Requires GPU for training

## Model Performance Summary 
| Model                        | Accuracy      | Precision (macro) | Recall (macro) | F1 (macro) | Notes                                   |
| ---------------------------- | ------------- | ----------------- | -------------- | ---------- | --------------------------------------- |
| Random Forest                | 0.54          | 0.54              | 0.54           | 0.53       | n_estimators=500, class_weight=balanced |
| SVM (RBF)                    | 0.55          | 0.54              | 0.55           | 0.54       | C=5, gamma=0.01                         |
| Logistic Regression          | 0.50          | 0.50              | 0.50           | 0.50       | max_iter=3000                           |
| **Wav2Vec2 (Deep Learning)** | **0.80–0.82** | **0.81**          | **0.79**       | **0.78**   | Fine-tuned, batch_size=16, lr=1e-5      |

**Cross-Validation (Random Forest 5-fold):** 0.541 ± 0.008  

**Conclusion:** Classical ML models provide a baseline, but their performance is limited by handcrafted features. Deep learning approaches (eg. Wav2Vec2) are needed to better capture complex audio patterns and improve accuracy.


## Azure Machine Learning Integration
**AutoML (Classical ML)**
- Uploaded CSV datasets to Azure ML Studio
- Task: Classification
- Target column: `label`
- Metrics tracked: Accuracy, Precision, Recall, AUC
- Best models registered for reuse

**Real-Time Deployment (Deep Learning)**
The Wav2Vec2 model is deployed using:
- Azure Managed Online Endpoint
- Custom inference code (`score.py`)
- Custom environment (`conda_env.yml`)
- Key-based authentication
- Public endpoint for real-time inference

## Key Learning Outcomes
- Audio signal processing & feature engineering
- Classical ML vs deep learning trade-offs
- Fine-tuning pretrained speech models
- Azure ML AutoML workflows
- Managed online endpoint deployment
- Debugging cloud inference environments
- Cost-aware ML system design
- Git & GitHub workflow for ML projects

  
## Technologies Used
- Python
- Librosa
- NumPy / Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers (Wav2Vec2)
- Azure Machine Learning (AutoML)
- LightGBM, RandomForest, Logistic Regression
- Git/GitHub

## References
- RAVDESS Dataset: [https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data)
- CREMA-D Dataset: [https://www.kaggle.com/datasets/ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad)
- Librosa: [https://github.com/librosa/librosa](https://github.com/librosa/librosa)
- Azure Machine Learning: [https://azure.microsoft.com/en-us/products/machine-learning](https://azure.microsoft.com/en-us/products/machine-learning)
