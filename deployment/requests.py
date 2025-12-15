import requests
import json
import numpy as np

# ----------------------------
# 1. Endpoint info
# ----------------------------
endpoint_url = "https://voice-emotion-ml-tptxt.eastasia.inference.ml.azure.com/score"
api_key = "YOUR_PRIMARY_KEY"  # replace with your primary key

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# ----------------------------
# 2. Define your emotion labels
# ----------------------------
labels = ['angry', 'fear', 'happy', 'neutral', 'sad']  # same as used in training

# ----------------------------
# 3. Create synthetic audio
# ----------------------------
synthetic_audio = np.random.randn(16000).astype(float).tolist()  # 1 second at 16kHz

# ----------------------------
# 4. Prepare payload
# ----------------------------
payload = json.dumps({
    "data": [synthetic_audio]
})

# ----------------------------
# 5. Send POST request
# ----------------------------
response = requests.post(endpoint_url, headers=headers, data=payload)

# ----------------------------
# 6. Get prediction and map to label
# ----------------------------
if response.status_code == 200:
    pred_idx = response.json()[0]  # Azure ML returns a list
    pred_label = labels[pred_idx]
    print("Predicted emotion:", pred_label)
else:
    print("Error:", response.status_code, response.text)
