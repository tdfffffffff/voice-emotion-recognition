import librosa
import numpy as np
import pandas as pd
import os

DATASET_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_data"
ROWS = []

for emotion in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for file in os.listdir(emotion_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(emotion_path, file)

        try:
            y, sr = librosa.load(file_path, duration=3)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1)

            row = mfcc_mean.tolist()
            row.append(emotion)
            ROWS.append(row)

        except Exception as e:
            print(f"Error processing {file}: {e}")

columns = [f"mfcc_{i}" for i in range(13)] + ["label"]
df = pd.DataFrame(ROWS, columns=columns)

output_path = "/Users/tdf/Downloads/voice_emotion_project/voice_emotion_features.csv"
df.to_csv(output_path, index=False)

print(f"Saved CSV to {output_path}")
print(df.head())

