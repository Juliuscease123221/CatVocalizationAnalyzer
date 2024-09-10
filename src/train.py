from utils import *
import torchaudio as ta
from torch.utils.data import WeightedRandomSampler
from torchsummary import summary
from constants import *
import torch as pt

if pt.cuda.is_available():
    device='cuda'
else:
    device='cpu'

def train_single_epoch(model,dataloader,loss_fn,optimizer,device):
    for waveform,label in (subPBar:=tqdm.tqdm(dataloader)):
        waveform=waveform.to(device)
        # label=pt.from_numpy(numpy.array(label))
        label=label.to(device)
        # calculate loss and preds
        logits=model(waveform)
        #loss=loss_fn(logits.float(),label.float().view(-1,1))
        loss=loss_fn(logits.float(),label)
        # backpropogate the loss and update the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        subPBar.set_description(f"loss:{loss.item()}")
    #print(f"loss:{loss.item()}")
    return loss.item()
    
def train(model,dataloader,loss_fn,optimizer,device,epochs):

    for i in (pbar:=tqdm.tqdm(range(epochs))):
        pbar.set_description(f"epoch:{i+1}")
        train_single_epoch(model,dataloader,loss_fn,optimizer,device)
        #print('-------------------------------------------')
    print('Finished Training')

#TODO
#Pull down the dataset from storage to the local FS

print(f"Creating MelSpectrogram Transformer")
melspectogram=ta.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=N_MELS)

print(f"Creating training dataset")
dataset=SoundDataset(
    audio_path="./src/cat_vocalization_dataset/",
    output_path="./src/generated_dataset/",
    label_path="./src/cat_vocalization_dataset/labels.json",
    transformation=melspectogram,
    target_sample_rate=TARGET_SAMPLE_RATE,
    num_samples=NUM_SAMPLES, #to be compatible with YAMNet
    slide_steps=WINDOW_SLIDE_STEPS, #The number of samples to slide the window for making multiple training audio data from a single audio file
    device=device
    )
print(f"Training dataset created of length: {len(dataset)} and with class counts {[x for x in zip(dataset.labels,dataset.class_counts)]}")

model=SoundCNN(label_count=len(dataset.labels))
summary(model,(1,N_MELS,32)) # number of channels, number of mels, (sample-rate/hop-length 16_000/512 = 31.25)

assert len(dataset.weights) == len(dataset), f"The len of weights ({len(dataset.weights)}) and the dataset items ({len(dataset)}) do not match."

#Setup our weighted Random Sampler
#https://saturncloud.io/blog/efficiently-sampling-batches-from-one-class-in-pytorch/#:~:text=To%20implement%20class%2Dbalanced%20sampling,being%20included%20in%20a%20batch.
#class_counts = dataset.class_counts
#class_weights = 1.0 / pt.Tensor(class_counts)
WRSampler = WeightedRandomSampler(weights=dataset.weights, num_samples=len(dataset))

# Old dataloader
# train_dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE,sampler=WRSampler)

#loss_fn=pt.nn.MSELoss()
loss_fn = pt.nn.CrossEntropyLoss()
optimizer=pt.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

train(model,train_dataloader,loss_fn,optimizer,device,EPOCHS)

#save Model weights
pt.save(model.state_dict(), SOUND_MODEL_SAVE_PATH)

#save labels as JSON array of strings
with open(SOUND_MODEL_LABELS_SAVE_PATH,"w") as F:
    F.write(str(dataset.labels))
