import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm  # progress bar

# Suppress Librosa warnings
warnings.filterwarnings("ignore", module="librosa")

# Paths
DATASET_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_data"
OUTPUT_CSV = "/Users/tdf/Downloads/voice_emotion_project/voice_emotion_features_enhanced.csv"

# Store rows
ROWS = []

# Loop over emotions (directories) with progress bar
emotions = [e for e in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, e))]
for emotion in tqdm(emotions, desc="Emotions"):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    # Loop over files with progress bar
    files = [f for f in os.listdir(emotion_path) if f.lower().endswith('.wav')]
    for file in tqdm(files, desc=f"Files ({emotion})", leave=False):
        file_path = os.path.join(emotion_path, file)

        try:
            # Load audio (short clips will be padded automatically by librosa)
            y, sr = librosa.load(file_path, sr=None, duration=3)
            if len(y) < 10:  # skip almost-empty clips
                continue

            # Dynamic FFT length
            n_fft = min(1024, len(y))

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1)

            # Chroma STFT
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            chroma_mean = chroma.mean(axis=1)

            # Spectral Contrast
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            spec_contrast_mean = spec_contrast.mean(axis=1)

            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = tonnetz.mean(axis=1)

            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = zcr.mean()

            # Combine features
            features = np.concatenate([
                mfcc_mean,
                chroma_mean,
                spec_contrast_mean,
                tonnetz_mean,
                [zcr_mean]
            ])

            # Append label
            row = features.tolist()
            row.append(emotion)
            ROWS.append(row)

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

# Column names
columns = (
    [f"mfcc_{i}" for i in range(13)] +
    [f"chroma_{i}" for i in range(12)] +
    [f"spec_contrast_{i}" for i in range(7)] +
    [f"tonnetz_{i}" for i in range(6)] +
    ["zcr"] +
    ["label"]
)

# Create DataFrame and save CSV
df = pd.DataFrame(ROWS, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nEnhanced feature CSV saved to: {OUTPUT_CSV}")
print(f"Total samples processed: {len(ROWS)}")
