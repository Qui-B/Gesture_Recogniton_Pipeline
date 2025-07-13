import os
import threading

import math
import time

import numpy as np
from torch import torch, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path


from src.PipelineModules.Classificator.GraphTCN import GraphTcn
from src.Config import (FRAMEWINDOW_LEN, TCN_NUM_OUTPUT_CLASSES, TCN_CHANNELS,
                        KERNEL_SIZE, GESTURE_SAMPLE_PATH, TCN_INPUT_SIZE, DROPOUT, BATCH_SIZE,
                        GCN_NUM_OUTPUT_CHANNELS, LEARNING_RATE_INIT, REL_PORTION_FOR_VALIDATION,
                        REL_PORTION_FOR_TESTING,
                        NUM_EPOCHS, DEVICE, SLEEP_BETWEEN_SAMPLES_S, LEARNING_RATE_STEPS, LEARNING_RATE_DECAY_FACTOR,
                        USE_ARTIFICIAL_SAMPLES, ARTIFICIAL_SAMPLES_PER_SAMPLE, USE_TRAINING_WEIGHTS, TRAINING_WEIGHTS)
from src.PipelineModules.Extractor.FeatureExtractor import FeatureExtractor

from src.Utility.Dataclasses import TrainingSample, FeaturePackage
from src.Utility.DebugManager.DebugManager import debug_manager
from src.Utility.Enums import Gesture
from src.Utility.Exceptions import WindowLengthException, UnsuccessfulCaptureException

#TODO implement threading for trainer, more method wrapping for the trainingprocess

def extract_window_from_mp4(feature_extractor, graph_TCN: GraphTcn, video_file):
    cap = cv2.VideoCapture(str(video_file))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_name = os.path.basename(video_file)
    if n_frames != FRAMEWINDOW_LEN:
        cap.release()
        raise WindowLengthException(n_frames, vid_name)
    else:
        print("Extracted: " + vid_name + " (" + str(n_frames) + "frames)") #DEBUG_SHOW_IMAGE

    feature_packages = []
    for frame_ind in range(0,n_frames):
        successful_read, frame = cap.read()
        if not successful_read:
            raise UnsuccessfulCaptureException

        feature_package = feature_extractor.extractStrat.extract(frame)
        if USE_TRAINING_WEIGHTS:
            feature_package = FeaturePackage(feature_package.lm_coordinates * TRAINING_WEIGHTS[frame_ind], feature_package.hand_detected) #TESTING TODO maybe remove

        # print("Vector: " + str(feature_package.lm_coordinates)) #TODO maybe implement as setting
        #
        # avg_abs_value = feature_package.lm_coordinates.abs().mean()
        # print("Mean change: " + str(avg_abs_value.item()))

        feature_packages.append(feature_package)
        debug_manager.show(frame)
        time.sleep(SLEEP_BETWEEN_SAMPLES_S)
    cap.release()
    return feature_packages

"""
Iterates over all subfolders of * corresponding to a gesture class. 
For each of these folders: extracts frame_deque for all mp4 files and labels it with the corresponding gesture class.

Returns:
    map: List of tuples, where each tuple contains a feature-frame_deque (deque) and a corresponding label (enum).
"""
def extract_training_data(training_Sample_path, graph_TCN, feature_extractor): #no multihreading support to for debug purposes
    path = Path(training_Sample_path)
    sample_dict = dict()
    print("Extract Datasamples...")

    for folder in path.iterdir():
        if not folder.is_dir() or folder.name not in Gesture.__members__: #filter out subDirectories that are not corresponding to a gesture
            continue
        print("Folder: \"" + folder.name + "\" found")
        cur_label = Gesture[folder.name] #frame_deque-label
        label_samples = []
        for video_file in folder.iterdir():
            if not video_file.is_file() or video_file.suffix not in ['.mp4','.avi', '.mov']: #filter out everything that is not a mp4 file
                continue
            try:
                feature_packages = extract_window_from_mp4(feature_extractor, graph_TCN, video_file)
                label_t = torch.tensor(cur_label.value, dtype=torch.long)
                cur_training_sample = TrainingSample(feature_packages, label_t)
                label_samples.append(cur_training_sample)

                if USE_ARTIFICIAL_SAMPLES:
                    for i in range(0, ARTIFICIAL_SAMPLES_PER_SAMPLE):
                        label_samples.append(cur_training_sample.applyNoise())

            except WindowLengthException as e:
                print(e.message)
        sample_dict[cur_label] = label_samples
    return sample_dict

def label_to_tensor(cur_label: Gesture):
    output_t = torch.full((TCN_NUM_OUTPUT_CLASSES,), -1, device=DEVICE) #No second dimension next to output classes to get a 1D tensor
    output_t[cur_label.value] = 1
    return output_t

def ran_elems_from_list(sample_list, num_elements):
    output_elems = []

    indices = np.random.choice(len(sample_list), size=num_elements,
                               replace=False)  # create random indices for fetching the test samples
    indices = sorted(indices, reverse=True)  # sorted in reverse order to ensure the indices are always inside the list
    for index in indices:
        output_elems.append(sample_list.pop(index))
    return output_elems

def split_training_data(sample_dict):
    training_list = []
    test_list = []
    validation_list = []

    for gesture_label, samples in sample_dict.items():
        cur_samples = list(samples)
        num_test_samples = math.floor(len(samples) * REL_PORTION_FOR_TESTING)
        num_validation_samples = math.floor(len(samples) * REL_PORTION_FOR_VALIDATION)

        test_list.extend(ran_elems_from_list(cur_samples, num_test_samples))
        validation_list.extend(ran_elems_from_list(cur_samples, num_validation_samples))
        training_list.extend(cur_samples)
    print("\nDataset split:")
    print("==========================================")
    print("{}% Test samples with length: {}".format((REL_PORTION_FOR_TESTING * 100), len(test_list)))
    print("{}% Validation samples with length: {}".format((REL_PORTION_FOR_VALIDATION*100), len(validation_list)))
    print("{}% Training samples with length: {}".format(((1-(REL_PORTION_FOR_TESTING+REL_PORTION_FOR_VALIDATION))*100), len(training_list)))

    return test_list,validation_list,training_list


class GestureDataset(Dataset):  #Wrapper class needed for using the dataloader
    def __init__(self, sample_list):
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    @staticmethod
    def collate_fn(batch): #needed to avoid batching Error with custom container-datatype (Trainingsample)
        return batch

def main() -> None:
    #Setup model
    model = GraphTcn(
        input_size = TCN_INPUT_SIZE,
        output_size = TCN_NUM_OUTPUT_CLASSES,
        gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
        num_channels_layer1 = TCN_CHANNELS,
        kernel_size = KERNEL_SIZE,
        dropout = DROPOUT)
    model.to(DEVICE)

    #Setup Feature Extractor (needed for producing the trainingsData)
    feature_extractor = FeatureExtractor(None,
                                         threading.Event(),
                                         threading.Event(),
                                         False,
                                         False) #samples already got filtered in the SampleCapturer

    #Setup Datasets
    test_data, validation_data, training_data  = split_training_data(extract_training_data(GESTURE_SAMPLE_PATH, feature_extractor, feature_extractor))

    test_dataset = GestureDataset(test_data)
    validation_dataset = GestureDataset(validation_data)
    training_dataset = GestureDataset(training_data)

    if len(test_dataset) > 0:
        test_dataLoader = DataLoader(test_dataset,
                                     1, #no batching as the testset is realtively small
                                     shuffle=True,#shuffle because datasets are still grouped in labels
                                     collate_fn=GestureDataset.collate_fn)
    if len(validation_dataset) > 0:
        validation_dataLoader = DataLoader(validation_dataset,
                                           BATCH_SIZE,
                                           shuffle=True,
                                           collate_fn=GestureDataset.collate_fn)

    training_dataLoader = DataLoader(training_dataset,
                                     BATCH_SIZE,
                                     shuffle=True,
                                     collate_fn=GestureDataset.collate_fn)


    #Setup training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_INIT)
    scheduler = StepLR(optimizer, step_size=LEARNING_RATE_STEPS, gamma=LEARNING_RATE_DECAY_FACTOR)  # for decaying learningrate

    #Start training
    #train_losses, val_losses, test_losses = [], [], [] #commented out because not needed for the current implementation but maybe in the future
    batch: list[TrainingSample] = []

    for epoch in range(NUM_EPOCHS):
        #training phase
        model.train()
        running_loss = 0.0
        for batch in training_dataLoader:
            optimizer.zero_grad()
            outputs = []
            labels = []

            for training_sample in batch: #Class: TrainingSample
                prediction = model(*training_sample.feature_packages)
                outputs.append(prediction)
                labels.append(training_sample.label)

            outputs = torch.cat(outputs).to(DEVICE)
            labels = torch.stack(labels).to(DEVICE)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(training_dataLoader.dataset)
        #train_losses.append(train_loss)
        scheduler.step()

        #validation phase
        model.eval()
        running_loss = 0.0
        outputs = []
        with torch.no_grad():
            for batch in validation_dataLoader:
                outputs = []
                labels = []

                for training_sample in batch:
                    prediction = model(*training_sample.feature_packages)
                    outputs.append(prediction)
                    labels.append(training_sample.label)

                outputs = torch.cat(outputs).to(DEVICE)
                labels = torch.stack(labels).to(DEVICE)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * len(labels)

        val_loss = running_loss / len(validation_dataLoader.dataset)
        #val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr'] #get current learning rate for outprint
        print(f"Epoch {epoch + 1} learning rate: {current_lr:.6f}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

    #final evaluation on test set
    running_loss = 0.0
    with torch.no_grad():
        for batch in test_dataLoader:
            outputs = []
            labels = []

            for training_sample in batch:
                prediction = model(*training_sample.feature_packages)
                outputs.append(prediction)
                labels.append(training_sample.label)

            outputs = torch.cat(outputs).to(DEVICE)
            labels = torch.stack(labels).to(DEVICE)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * len(labels)

        test_loss = running_loss / len(test_dataLoader.dataset)
        #test_losses.append(test_loss)
        print(f"Final loss on the test set: {test_loss:.4f}")


    torch.save(model.state_dict(), "../PipelineModules/Classificator/trained_weights.pth")

if __name__ == "__main__":
    main()