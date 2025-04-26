from collections import deque
import torch
import cv2
from pathlib import Path

from Enums import Gesture
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.Classificator.TCN_SinglePool import TCN_SP
from PipelineModules.FeatureExtractor import FeatureExtractor
from Settings import WINDOW_LENGTH, NUM_OUTPUT_CLASSES, NUM_CHANNELS_LAYER2, NUM_CHANNELS_LAYER1, \
    KERNEL_SIZE, GESTURE_SAMPLE_PATH, FEATURE_VECTOR_LENGTH
from Exceptions import WindowLengthException
from torch.utils.data import DataLoader, Dataset

from Hand_Recognition.Settings import BATCH_SIZE

"""
Extracts a feature window from a given video file using a feature extractor and window manager.

Args: 
    feature_extractor: used to extract the features from every frame (FeatureExtractor.py)
    window_manager: used to combine the features from every frame to a window-sequence (WindowManager.py)
    video_file: path to video file

Returns:
    deque: sequence of feature_vectors
    
Raises:
    WindowLengthException: If the number of video-frames differs from the length of the window (Settings.WINDOW_LENGTH)
"""
def extractWindowFromMP4(feature_extractor, graph_TCN: GraphTcn,  video_file):
    cap = cv2.VideoCapture(str(video_file))
    frame_count = 0
    while True:
        successful_read, frame = cap.read()
        if not successful_read: #Case: End of video
            break
        frame_count += 1
        feature_package = feature_extractor.extract(frame)
        graph_TCN.updateWindow(feature_package)

    if frame_count != WINDOW_LENGTH:
        cap.release()
        raise WindowLengthException

    cap.release()
    return deque(graph_TCN.getWindowTensor())

"""
Iterates over all subfolders of * corresponding to a gesture class. 
For each of these folders: extracts window for all mp4 files and labels it with the corresponding gesture class.

Returns:
    list: List of tuples, where each tuple contains a feature-window (deque) and a corresponding label (enum).
"""
def extractTrainingData(training_Sample_path, graph_TCN, feature_extractor):
    path = Path(training_Sample_path)
    sample_list = [] #Holds tuples (window,gesture-label)

    for folder in path.iterdir():
        if not folder.is_dir() or folder.name not in Gesture.__members__: #filter out subDirectories that are not corresponding to a gesture
            continue
        print("folder: " + folder.name + " found")
        cur_label = Gesture[folder.name] #window-label

        for video_file in folder.iterdir():
            if not video_file.is_file() or video_file.suffix not in ['.mp4']: #filter out everything that is not a mp4 file
                continue
            try:
                window = extractWindowFromMP4(feature_extractor, graph_TCN, video_file)
                sample_list.append((window, cur_label))
            except WindowLengthException as e:
                print(e.message)

    return sample_list


class GestureDataset(Dataset):  #Wrapper class needed for using the dataloader
    def __init__(self, training_sample_path, graph_TCN, feature_extractor):
        self.sample_list = extractTrainingData(training_sample_path, graph_TCN, feature_extractor)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]

#Check if pc has a GPU on board to speed up the training process
if torch.cuda.is_available():
    device = torch.device("cuda:0") #In case of multiple GPU's the first one gets used indicated by 0
else:
    device = torch.device("cpu")

#Setup model
gesture_classifier = GraphTcn(
    input_size = WINDOW_LENGTH * FEATURE_VECTOR_LENGTH,
    output_size = NUM_OUTPUT_CLASSES,
    num_channels_layer1 = NUM_CHANNELS_LAYER1,
    num_channels_layer2 = NUM_CHANNELS_LAYER2,
    kernel_size = KERNEL_SIZE)

#Setup Feature Extractor (needed for producing the trainingsData)
feature_extractor = FeatureExtractor()

#Setup Dataset
dataset = GestureDataset(GESTURE_SAMPLE_PATH, gesture_classifier, feature_extractor)
dataLoader = DataLoader(dataset, BATCH_SIZE,  shuffle=True)

