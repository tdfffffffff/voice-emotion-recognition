# voice-emotion-recognition

## Overview
This project is a **voice emotion recognition system** built using public audio datasets (RAVDESS, CREMA-D) to extract features and train machine learning models to predict emotions from voice recordings.

**The pipeline**:

1. Extracts and aggregates audio features (MFCCs, Chroma, Spectral Contrast, Tonnetz, Zero-Crossing Rate)
2. Converts features into tabular CSV data
3. Trains multiple classification models using **Microsoft Azure AutoML**
4. Predicts emotions like `happy`, `sad`, `angry`, `fear`, `neutral`
   
## Key Features

- **Audio feature extraction**: MFCC, Chroma STFT, Spectral Contrast, Tonnetz, Zero-Crossing Rate
- **Classical ML models** trained on CSV datasets using Azure ML AutoML
- **Large file support** via Git LFS for audio and CSV files
- **Cost-efficient pipeline**: CPU-based ML models with feature engineering instead of GPU-heavy deep learning
- **Optimised accuracy**:  
  - **Baseline**: ~55%
  - **Enhanced**: ~65–70% with feature engineering and model selection

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
- Azure Machine Learning (AutoML)
- LightGBM, RandomForest, Logistic Regression (Auto-selected)

## References
- RAVDESS Dataset: [https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data](url)
- CREMA-D Dataset: [https://www.kaggle.com/datasets/ejlok1/cremad](url)
- Librosa: [https://github.com/librosa/librosa](url)
- Azure Machine Learning: [https://azure.microsoft.com/en-us/products/machine-learning](url)
