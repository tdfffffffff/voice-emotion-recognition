import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", module="librosa")

DATASET_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_data"
OUTPUT_CSV = "/Users/tdf/Downloads/voice_emotion_project/voice_emotion_features_max.csv"
ROWS = []

for emotion in tqdm(os.listdir(DATASET_PATH), desc="Processing emotions"):
    emotion_path = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for file in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file)
        if not file_path.lower().endswith('.wav'):
            continue

        try:
            y, sr = librosa.load(file_path, sr=None, duration=3)
            if len(y) < 10:
                continue

            n_fft = min(1024, len(y))

            # MFCC + delta + delta-delta
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_features = np.concatenate([mfcc.mean(axis=1),
                                            mfcc_delta.mean(axis=1),
                                            mfcc_delta2.mean(axis=1)])

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
            chroma_mean = chroma.mean(axis=1)

            # Spectral contrast
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
            spec_contrast_mean = spec_contrast.mean(axis=1)

            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = tonnetz.mean(axis=1)

            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y).mean()

            # RMS energy
            rms = librosa.feature.rms(y=y).mean()

            # Spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

            # Combine features
            features = np.concatenate([
                mfcc_features,
                chroma_mean,
                spec_contrast_mean,
                tonnetz_mean,
                [zcr, rms, spec_bw]
            ])

            row = features.tolist()
            row.append(emotion)
            ROWS.append(row)

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

# Column names
columns = (
    [f"mfcc_{i}" for i in range(13*3)] +  # MFCC + delta + delta-delta
    [f"chroma_{i}" for i in range(12)] +
    [f"spec_contrast_{i}" for i in range(7)] +
    [f"tonnetz_{i}" for i in range(6)] +
    ["zcr", "rms", "spec_bw"] +
    ["label"]
)

df = pd.DataFrame(ROWS, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Enhanced CSV saved to: {OUTPUT_CSV}, total samples: {len(ROWS)}")
