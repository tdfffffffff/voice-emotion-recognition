import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

def init():
    global model, processor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("emotion_model")
    processor = Wav2Vec2Processor.from_pretrained("emotion_model")
    model.to(device)
    model.eval()

def run(raw_data):
    # raw_data: list of audio arrays (numpy)
    input_values = processor(raw_data, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
        preds = torch.argmax(logits, dim=-1)
    return preds.cpu().tolist()
