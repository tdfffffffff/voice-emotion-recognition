import librosa
import numpy as np
from score import init, run
from sklearn.preprocessing import LabelEncoder

# Initialize model
init()

# Define emotion classes
le = LabelEncoder()
le.classes_ = np.array(['angry', 'fear', 'happy', 'neutral', 'sad'])

# Option 1: Test with a real audio file
# audio_path = "voice_emotion_project/happy/sample.wav"
# waveform, sr = librosa.load(audio_path, sr=16000)
# pred_idx = run([waveform])
# pred_label = le.inverse_transform(pred_idx)
# print("Predicted emotion:", pred_label)

# Option 2: Test with synthetic audio
synthetic_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
pred_idx = run([synthetic_audio])
pred_label = le.inverse_transform(pred_idx)
print("Predicted emotion (synthetic):", pred_label)
