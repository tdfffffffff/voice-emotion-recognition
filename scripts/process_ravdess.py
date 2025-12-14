import os
import shutil

RAVDESS_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_raw/ravdess"
OUTPUT_PATH = "/Users/tdf/Downloads/voice_emotion_project/audio_data"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

for actor in os.listdir(RAVDESS_PATH):
    actor_path = os.path.join(RAVDESS_PATH, actor)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        parts = file.split("-")
        if len(parts) > 2:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                src = os.path.join(actor_path, file)
                dst = os.path.join(OUTPUT_PATH, emotion, file)
                shutil.copy(src, dst)
