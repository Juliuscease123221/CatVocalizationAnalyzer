import torch as pt
from torch import nn
import torchaudio as ta
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader

import numpy as numpy

import os
import glob
import tqdm as tqdm
import json
import shutil

class SoundCNN(nn.Module):

    def __init__(self,label_count):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        #for self.linear1
        #in_feature count needs to be adjusted based on the mel spec input data shape
        #which depends on the number of mels and the sample-rate/hop-length value
        
        #64 mels, 16000 target sample rate and 512 hop length
        #  self.linear1=nn.Linear(in_features=128*5*3,out_features=128) 
        
        #128 mels
        self.linear1=nn.Linear(in_features=128*9*3,out_features=128)

        self.linear2=nn.Linear(in_features=128,out_features=label_count) 
        self.output=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        logits=self.linear2(x)
        output=self.output(logits)
        
        return output

class SoundDataset(Dataset):

    '''
    Assumes a data set has been downlaoded to a local folder
    will itterate through one layer of sub folders, assuming the sub-folder names are the labels
    files with sample counts larger than the sample window will be sub divided into multiple data files
    new data files are stored in a different dir using the orginal file names as the base names
    each newly generated data file and 
    '''
    def __init__(self,audio_path,output_path,label_path,transformation,target_sample_rate,num_samples,slide_steps,device):

        self.audio_path=audio_path
        self.output_path=output_path
        self.label_path=label_path
        self.device=device
        self.transformation=transformation.to(device)
        self.target_sample_rate=target_sample_rate
        self.num_samples=num_samples
        self.slide_steps=slide_steps
        self.datalist = []
        self.labels = []
        self.class_counts = []
        self.weights = []

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)
        listOfWaveFilePaths = glob.glob(f"{audio_path}/*/*.wav")

        with open(label_path) as labelsFile:
            self.labels = json.load(labelsFile)
            
            if type(self.labels) != list:
                raise Exception(f"labels in labels file must be a list array")

            for label in self.labels:
                if type(label) not in [str]:
                    raise Exception(f"labels in labels file must be of type string")
                
            #initialize the class count list to 0 for each label
            self.class_counts = [0 for x in range(len(self.labels))]

        for filePath in listOfWaveFilePaths:
            if filePath.endswith('wav'):

                #get the label by getting the sub directory the file is in
                label = filePath.split("/")[-2]

                if label not in self.labels:
                    print(f"Skipping the data processing of {filePath} as {label} is not in the list of labels")
                    continue

                #get filename without extention
                dirPath, filename = os.path.split(filePath)
                #basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
                basename_without_ext, _ = os.path.splitext(filename)

                #determine if the wav file is bigger than the window
                #Get the waveform samples and the sample rate
                waveform, sr = ta.load(filePath)

                #upsampel of downsample now
                waveform = self._resample(waveform,sr)

                #set counter to 0 for while loop
                counter = 0

                #inc over the waveform in sliding window steps
                #process the waveform if it is longer than a sample window minus the step, 
                #what's missing will be padded later in the process
                while waveform.shape[1] > (self.num_samples - self.slide_steps):                        
                    newFileName = f"{basename_without_ext}_{counter:02d}.wav"
                    outFilePath = os.path.join(self.output_path,newFileName)

                    #set the output waveform to the first window in the source waveform
                    outWaveform = waveform[:,:self.num_samples]

                    #save the outWaveform at the target sample rate
                    ta.save(outFilePath,outWaveform,self.target_sample_rate)

                    #inc the class counter for the label
                    self.class_counts[self.labels.index(label)] += 1

                    #add out file path to datalist with the label string
                    self.datalist.append((outFilePath,label))

                    #chomp off half a sample window from the beggining of the waveform and then repeat
                    waveform=waveform[:,self.slide_steps:]
                    counter+=1
                
        #calculate the weights
        for item in self.datalist:
            classLabelIndex = item[1]
            self.weights.append(1/self.class_counts[self.labels.index(classLabelIndex)])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self,idx):
        audio_file_path,label = self.datalist[idx]
        waveform,sample_rate=ta.load(audio_file_path) #(num_channels,samples) -> (1,samples) makes the waveform mono
        waveform=waveform.to(self.device)
        waveform=self._resample(waveform,sample_rate)   
        waveform=self._mix_down(waveform)
        waveform=self._cut(waveform)
        waveform=self._right_pad(waveform)
        waveform=self.transformation(waveform)

        labelIndex = self.labels.index(label)

        return waveform, labelIndex
      
    def _resample(self,waveform,sample_rate):
        # used to handle sample rate
        resampler=ta.transforms.Resample(sample_rate,self.target_sample_rate)
        return resampler(waveform)
    
    def _mix_down(self,waveform):
        # used to handle channels
        waveform=pt.mean(waveform,dim=0,keepdim=True)
        return waveform
    
    def _cut(self,waveform):
        # cuts the waveform if it has more than certain samples
        if waveform.shape[1]>self.num_samples:
            waveform=waveform[:,:self.num_samples]
        return waveform
    
    def _right_pad(self,waveform):
        # pads the waveform if it has less than certain samples
        signal_length=waveform.shape[1]
        if signal_length<self.num_samples:
            num_padding=self.num_samples-signal_length
            last_dim_padding=(0,num_padding) # first arg for left second for right padding. Make a list of tuples for multi dim
            waveform=pt.nn.functional.pad(waveform,last_dim_padding)
        return waveform


#model=CNNNetwork().cuda()
#summary(model,(1,64,44))
