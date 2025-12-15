# voice-emotion-recognition

This repository contains a **voice emotion recognition system** built using public audio datasets (RAVDESS, CREMA-D). The project demonstrates both **classical machine learning on handcrafted features** and **deep learning using Wav2Vec2** to predict emotions from voice recordings.

## Overview
The project explores the trade-offs between CPU-efficient classical ML pipelines and GPU-powered deep learning models:

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
     
## Key Features
- Audio feature extraction: MFCC, Chroma STFT, Spectral Contrast, Tonnetz, Zero-Crossing Rate, RMS  
- Classical ML models trained on CSV datasets using Azure ML AutoML  
- Deep learning Wav2Vec2 notebook for end-to-end raw audio classification  
- Large file support via Git LFS for audio and CSV files  
- Cost-efficient design: CPU-only classical ML models vs GPU deep learning  
- Optimised accuracy:  
  - Classical ML baseline: ~55%  
  - Enhanced with feature engineering: ~65–70%  
  - Deep learning Wav2Vec2: ~80–72%  

## Model Performance

| Model                 | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-score (macro avg) | Notes / Hyperparameters |
|-----------------------|----------|---------------------|------------------|---------------------|------------------------|
| Random Forest         | 0.54     | 0.54                | 0.54             | 0.53                | n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced' |
| SVM (RBF Kernel)      | 0.55     | 0.54                | 0.55             | 0.54                | C=5, gamma=0.01, class_weight='balanced' |
| Logistic Regression   | 0.50     | 0.50                | 0.50             | 0.50                | max_iter=3000, class_weight='balanced' |
| Wav2Vec2 (Deep Learning) | 0.82     | 0.81                | 0.79             | 0.78                | Pretrained Wav2Vec2 base, fine-tuned on raw audio, batch_size=16, learning_rate=1e-5, epochs=10 |

**Cross-Validation (Random Forest 5-fold):** 0.541 ± 0.008  

> **Conclusion:** Classical ML models provide a baseline, but their performance is limited by handcrafted features. Deep learning approaches (eg. Wav2Vec2) are needed to better capture complex audio patterns and improve accuracy.


## Key Learning Outcomes
- Audio signal processing and feature engineering
- Classical ML vs deep learning trade-offs
- Azure ML AutoML workflows
- Model evaluation using accuracy and weighted AUC
- Cost-aware ML system design (CPU-only)

## Azure ML Integration

1. Upload `voice_emotion_features.csv` or `voice_emotion_features_enhanced.csv` to Azure ML Studio
2. Create an **Automated ML Classification Job**:
   - Task type: `Classification`
   - Target column: `label`
   - Validation type: Train-validation split
   - Compute: Compute instance (small VM)
   - Track metrics: Accuracy, Precision, Recall, AUC
3. Register the best model for predictions

  
## Technologies Used
- Python
- Librosa
- NumPy / Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers (Wav2Vec2)
- Azure Machine Learning (AutoML)
- LightGBM, RandomForest, Logistic Regression

## References
- RAVDESS Dataset: [https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data)
- CREMA-D Dataset: [https://www.kaggle.com/datasets/ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad)
- Librosa: [https://github.com/librosa/librosa](https://github.com/librosa/librosa)
- Azure Machine Learning: [https://azure.microsoft.com/en-us/products/machine-learning](https://azure.microsoft.com/en-us/products/machine-learning)
