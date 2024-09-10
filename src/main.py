import os, json
from flask import Flask, request, jsonify 
from flask_sslify import SSLify
from google.cloud import storage
from utils import SoundCNN
import torch as pt
#import numpy as np
#from datetime import datetime, timezone
from constants import *
import random
import uuid
from utils import *
import json
import shutil

print("Starting Sound Classification and Regressor (CAR) service ...")

storage_client = storage.Client(project='catverse-poc')
modelsBucket = storage_client.bucket(MODELS_BUCKET)
soundCNN = None
device = None
modelLabels = []
MODEL_TO_LOAD_PATH = "./models/sound_classifier_230926.pt"
MODEL_LABELS_TO_LOAD_PATH = "./models/sound_classifier_230926.json"


#create and load 
try:

    if pt.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    #check if local saved model state dict exsists
    #if not download it
    #if all esle fails, throw an exception
    
    #TODO download model and labels from bucket
    #load model labels
    with open(MODEL_LABELS_TO_LOAD_PATH) as F:
        modelLabels = json.load(F)

    soundCNN = SoundCNN(len(modelLabels))
    soundCNN.load_state_dict(pt.load(MODEL_TO_LOAD_PATH))
    soundCNN.eval()


except Exception as e:
    text = f"Got excpetion while loading model from saved dictionary: {str(e)}"
    print(text)
    raise Exception(text)

print(f"Creating MelSpectrogram Transformer")
melspectogram=ta.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=N_MELS)

#create the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = "change me"

@app.route('/healthcheck',methods=['GET'])
def health_check():

    _response = {"message":"Service is up!"}

    return (jsonify(_response),200)

@app.route('/api/poc/soundClassifier',methods=['PUT'])
def soundClassifier():

    print(f"soundClassifier: Entering")

    results = {}

    #get the sound file
    print(f"soundClassifier: Checking for wav file payload")
    if not request.data:
        message = "Error, no wav file payload found"
        print(f"soundClassifier: {message}")
        return (jsonify({"message":message}),400)
    
    #TODO validate the payload is a wav file

    print(f"soundClassifier: payload recieved")

    #Save wav file to local folder with temp name
    #setup a folder structure the way utils.SoundDataset expects the files to be with a sub dir for each label
    tempDirName = f"{str(uuid.uuid4())}"
    tempDirPath = os.path.join(TEMP_SOUND_FILE_PATH,tempDirName)
    tempDirInPath = os.path.join(tempDirPath,"in")
    tempDirGenPath = os.path.join(tempDirPath,"gen")
    tempDirPathUnknown = os.path.join(tempDirInPath,"unknown")

    print(f"soundClassifier: create dir {tempDirPathUnknown}")
    #os.mkdir(os.path.join(TEMP_SOUND_FILE_PATH,tempDirName))
    os.makedirs(tempDirPathUnknown)

    if not os.path.isdir(tempDirPathUnknown):
        raise Exception(f"Couldn't create dir: {tempDirPathUnknown}")

    tempFileName = "input.wav"
    inputFilePath = os.path.join(tempDirPathUnknown,tempFileName)

    #write the audio file as bytes
    with open(inputFilePath,"wb") as TF:
        TF.write(request.data)

    if not os.path.isfile(inputFilePath):
        raise Exception(f"{inputFilePath} was not created")

    tempLabelsFileName = "input.json"
    tempLabelsPath = os.path.join(tempDirInPath,tempLabelsFileName)

    #write the json labels file needed by SoundDataset
    with open(tempLabelsPath,"w") as TF:
        TF.write('["unknown"]')

    if not os.path.isfile(tempLabelsPath):
        raise Exception(f"{tempLabelsPath} was not created")

    #Create a sound Dataset
    print(f"Creating sound dataset for reqeusted audio file")
    dataset=SoundDataset(
        audio_path=tempDirInPath,
        output_path=tempDirGenPath,
        label_path=tempLabelsPath,
        transformation=melspectogram,
        target_sample_rate=TARGET_SAMPLE_RATE,
        num_samples=NUM_SAMPLES, #to be compatible with YAMNet
        slide_steps=WINDOW_SLIDE_STEPS, #The number of samples to slide the window for making multiple training audio data from a single audio file
        device=device
    )
    print(f"Sound dataset created of length: {len(dataset)} and with class counts {[x for x in zip(dataset.labels,dataset.class_counts)]}")

    # call the cat vocalziation classifier on all items in gen dataset 
    # and average the scores
    labelScoreCounters = [0 for x in modelLabels]

    print(f"soundClassifier: trying soundCNN.inference() ...")
    try:
        counter = 0
        for waveform, label_index in dataset:
            soundCNN.eval()
            inputs = pt.unsqueeze(waveform,0)
            with pt.no_grad():
                predictions=soundCNN(inputs)
            #add the label scores to the coutners
            #need to convert the Tensor object to a list
            predictionsList = predictions[0].tolist()
            print(f"soundClassifier: predictions {counter:02d}: {predictionsList}")
            labelScoreCounters = list(map(lambda x,y: x+y, labelScoreCounters,predictionsList))
            counter += 1

    except Exception as e:
        errorMessage = f"Got unhandled exception: {str(e)}"
        return (jsonify(errorMessage),400)

    #Delete the temp dir for the requested audio file
    print(f"Deleting temp dir {tempDirPath}")
    if os.path.exists(tempDirPath):
        shutil.rmtree(tempDirPath)

    #create the scores list by averaging the totals based on the dataset length
    scores = [x/len(dataset) for x in labelScoreCounters]

    results["labels"] = modelLabels #TODO replace with labels
    results["scores"] = scores

    print(f"soundClassifier: results: {results}")
    print(f"soundClassifier: Exiting")

    return (jsonify(results), 200)
