from typing import NamedTuple
from PipelineModules.DataClasses import FeaturePackage, SpatialFeaturePackage
import math
import numpy as np
from torch import torch, nn
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path

from Enums import Gesture
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.FeatureExtractor import FeatureExtractor
from Config import (WINDOW_LENGTH, NUM_OUTPUT_CLASSES, NUM_CHANNELS_LAYER2, NUM_CHANNELS_LAYER1,
                    KERNEL_SIZE, GESTURE_SAMPLE_PATH, FEATURE_VECTOR_LENGTH, INPUT_SIZE, DROPOUT, BATCH_SIZE,
                    GCN_NUM_OUTPUT_CHANNELS, LEARNING_RATE, REL_PORTION_FOR_VALIDATION, REL_PORTION_FOR_TESTING,
                    NUM_EPOCHS, DEVICE)
from Exceptions import WindowLengthException, UnsuccessfulCaptureException

class TrainingSample(NamedTuple):
        feature_packages: list[FeaturePackage]
        label: torch.Tensor


#Rewrite to fit the forward method
def extractWindowFromMP4(feature_extractor, graph_TCN: GraphTcn,  video_file):
    cap = cv2.VideoCapture(str(video_file))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames != WINDOW_LENGTH:
        cap.release()
        raise WindowLengthException

    feature_packages = []
    for frame_ind in range(0,n_frames):
        successful_read, frame = cap.read()
        if not successful_read:
            raise UnsuccessfulCaptureException

        feature_packages.append(feature_extractor.extract(frame))

    cap.release()
    return feature_packages

"""
Iterates over all subfolders of * corresponding to a gesture class. 
For each of these folders: extracts frame_deque for all mp4 files and labels it with the corresponding gesture class.

Returns:
    map: List of tuples, where each tuple contains a feature-frame_deque (deque) and a corresponding label (enum).
"""
def extractTrainingData(training_Sample_path, graph_TCN, feature_extractor):
    path = Path(training_Sample_path)
    sample_dict = dict()

    for folder in path.iterdir():
        if not folder.is_dir() or folder.name not in Gesture.__members__: #filter out subDirectories that are not corresponding to a gesture
            continue
        print("folder: " + folder.name + " found")
        cur_label = Gesture[folder.name] #frame_deque-label
        label_samples = []
        for video_file in folder.iterdir():
            if not video_file.is_file() or video_file.suffix not in ['.mp4']: #filter out everything that is not a mp4 file
                continue
            try:
                feature_packages = extractWindowFromMP4(feature_extractor, graph_TCN, video_file)
                label_t = labelToTensor(cur_label)
                cur_training_sample = TrainingSample(feature_packages, label_t)
                label_samples.append(cur_training_sample)
            except WindowLengthException as e:
                print(e.message)
        sample_dict[cur_label] = label_samples

    return sample_dict

def labelToTensor(cur_label: Gesture):
    output_t = torch.full((NUM_OUTPUT_CLASSES,) , -1, device=DEVICE) #No second dimension next to output classes to get a 1D tensor
    output_t[cur_label.value] = 1
    return output_t

def ranElemsFromList(sample_list, num_elements):
    output_elems = []

    indices = np.random.choice(len(sample_list), size=num_elements,
                               replace=False)  # create random indices for fetching the test samples
    indices = sorted(indices, reverse=True)  # sorted in reverse order to ensure the indices are always inside the list
    for index in indices:
        output_elems.append(sample_list.pop(index))
    return output_elems

def splitTrainingData(sample_dict):
    training_list = []
    test_list = []
    validation_list = []

    for gesture_label, samples in sample_dict.items():
        cur_samples = list(samples)
        num_test_samples = math.floor(len(samples) * REL_PORTION_FOR_TESTING)
        num_validation_samples = math.floor(len(samples) * REL_PORTION_FOR_VALIDATION)

        test_list.extend(ranElemsFromList(cur_samples, num_test_samples))
        validation_list.extend(ranElemsFromList(cur_samples, num_validation_samples))
        training_list.extend(cur_samples)
    print("\nDataset split:")
    print("==========================================")
    print("{}% Test samples with length: ".format((REL_PORTION_FOR_TESTING * 100), len(validation_list)))
    print("{}% Validation samples with length: ".format((REL_PORTION_FOR_VALIDATION*100), len(validation_list)))
    print("{}% Training samples with length: ".format(((1-(REL_PORTION_FOR_TESTING+REL_PORTION_FOR_VALIDATION))*100), len(training_list)))

    return test_list,validation_list,training_list


class GestureDataset(Dataset):  #Wrapper class needed for using the dataloader
    def __init__(self, sample_list):
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]

"""
#Check if pc has a GPU on board to speed up the training process
if torch.cuda.is_available():
    device = torch.device("cuda:0") #In case of multiple GPU's the first one gets used indicated by 0
else:
    device = torch.device("cpu")
"""

#Setup model
model = GraphTcn(
    input_size = INPUT_SIZE,
    output_size = NUM_OUTPUT_CLASSES,
    gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
    num_channels_layer1 = NUM_CHANNELS_LAYER1,
    num_channels_layer2 = NUM_CHANNELS_LAYER2,
    kernel_size = KERNEL_SIZE,
    dropout = DROPOUT)

#Setup Feature Extractor (needed for producing the trainingsData)
feature_extractor = FeatureExtractor()

#Setup Datasets
test_data, validation_data, training_data  = splitTrainingData(extractTrainingData(GESTURE_SAMPLE_PATH, feature_extractor, feature_extractor))

test_dataset = GestureDataset(test_data)
validation_dataset = GestureDataset(validation_data)
training_dataset = GestureDataset(training_data)


test_dataLoader = DataLoader(test_dataset, BATCH_SIZE,  shuffle=True)
validation_dataLoader = DataLoader(validation_dataset, BATCH_SIZE,  shuffle=True)
training_dataLoader = DataLoader(training_dataset, BATCH_SIZE,  shuffle=True) #shuffle because datasets are still grouped in labels

#Setup training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Start training
train_losses, val_losses = [], []
batch: list[TrainingSample] = []

for epoch in range(NUM_EPOCHS):
    #training phase
    model.train()
    running_loss = 0.0
    for batch in training_dataLoader: #only one sample in this case as BATCH_SIZE is 1
        optimizer.zero_grad()
        outputs = []
        labels = []
        for training_sample in batch: #Class: TrainingSample
            prediction = model(training_sample.feature_packages)
            outputs.append(prediction)
            labels.append(training_sample.label)
        outputs = torch.tensor(outputs, device=DEVICE)
        labels = torch.tensor(labels, device=DEVICE)
        loss = criterion(torch.tensor(outputs),torch.tensor(labels))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    train_loss = running_loss / len(training_dataLoader.dataset)
    train_losses.append(train_loss)
    #validation phase
    model.eval()
    running_loss = 0.0
    outputs = []
    with torch.no_grad():
        for batch in validation_dataLoader:
            outputs = []
            labels = []
            for training_sample in batch:  # Class: TrainingSample
                prediction = model(training_sample.feature_packages)
                outputs.append(prediction)
                labels.append(training_sample.label)
            outputs = torch.tensor(outputs, device=DEVICE)
            labels = torch.tensor(labels, device=DEVICE)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * len(labels)
    val_loss = running_loss / len(validation_dataLoader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train loss: {train_loss}, Validation loss: {val_loss}")
torch.save(model.state_dict(), "trained_model.pth")
