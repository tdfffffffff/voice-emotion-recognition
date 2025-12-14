import os
import shutil

CREMAD_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_raw/cremad"
OUTPUT_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_data"

emotion_map = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear"
}

for file in os.listdir(CREMAD_PATH):
    if not file.endswith(".wav"):
        continue
    for code, emotion in emotion_map.items():
        if f"_{code}_" in file:
            src = os.path.join(CREMAD_PATH, file)
            dst = os.path.join(OUTPUT_PATH, emotion, file)
            shutil.copy(src, dst)
            break
