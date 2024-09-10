
import torch as pt
import torchaudio as ta
from utils import SoundCNN, SoundDataset
from constants import *
import json

STATE_DIC_PATH = "./models/sound_classifier_230926.pt"
STATE_DIC_LABELS_PATH = "./models/sound_classifier_230926.json"

if pt.cuda.is_available():
    device='cuda'
else:
    device='cpu'

#load saved model labels
with open(STATE_DIC_LABELS_PATH) as LabelsFile:
    savedModelLabels = json.load(LabelsFile)

melspectogram=ta.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=N_MELS)

dataset=SoundDataset(
    audio_path="./datasets/test_dataset/",
    output_path="./datasets/test_generated_dataset/",
    label_path="./datasets/test_dataset/labels.json",
    transformation=melspectogram,
    target_sample_rate=TARGET_SAMPLE_RATE,
    num_samples=NUM_SAMPLES, #to be compatible with YAMNet should be set to 16kHz and 0.975s
    slide_steps=WINDOW_SLIDE_STEPS, #The number of samples to slide the window for making multiple training audio data from a single audio file
    device=device
    )

model = SoundCNN(label_count=len(savedModelLabels))
model.load_state_dict(pt.load(STATE_DIC_PATH))
model.eval()


def predict(model,inputs):
    model.eval()
    inputs=pt.unsqueeze(inputs,0)
    with pt.no_grad():
        predictions=model(inputs)
    
    return predictions

print(f"Labels: {savedModelLabels}")
for waveform,label_index in dataset:
    predictions=predict(model,waveform)
    #print the Tensor output as a list
    print(f"predictions: {predictions[0].tolist()} - Predicted Label: {savedModelLabels[pt.argmax(predictions, axis=1)]} - Actual Label: {dataset.labels[label_index]}")