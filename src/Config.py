import os

from sympy import false
from torch import torch,nn

#Collection of all setting-parameters used by the pipeline modules


#===========================
#Performance
#===========================
SKIP_N_FRAMES = 0 #drop frames inbetween for more consistency


EXTRACTOR_NUM_THREADS = 2 #Mainly used by the mediapipe extraction
#Only in case of performance problems on CPU
CLASSIFICATOR_NUM_THREADS = 8
CLASSIFICATOR_NUM_INTEROP_THREADS = 2
#===========================
#FrameCapturer
#===========================
IMAGE_SOURCE = 1 #droid cam: 0 webcam: 1
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


#===========================
#LmExtractor
#===========================
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FEATURE_VECTOR_LENGTH = 21 #number of landmarks
FEATURE_VECTOR_WIDTH = 3 #number of coordinates per landmark


#===========================
#FrameWindow
#===========================
WINDOW_LENGTH = 31

#===========================
#Classificator
#===========================
#FEATURE_VECTOR_LENGTH and WINDOW_LENGTH also get used for input_size
KERNEL_SIZE = 3
NUM_OUTPUT_CLASSES = 7 #scroll up/down, swipe left/right, zoom in/out, nothing
GCN_NUM_OUTPUT_CHANNELS = 36
INPUT_SIZE = FEATURE_VECTOR_LENGTH * GCN_NUM_OUTPUT_CHANNELS + 1 #+1 because of the HAND_DETECTED feature
DROPOUT = 0.1 #TESTING
DEVICE = 'cpu' # or cuda:0

#Channels for the tcn layers
NUM_CHANNELS_LAYER1 = [WINDOW_LENGTH,WINDOW_LENGTH,WINDOW_LENGTH,WINDOW_LENGTH] #4 Layers because it allows us to get a receptive window of 31
NUM_CHANNELS_LAYER2 = [16,16,16,16]
POOLSTRIDE = 2


#Edge index for the gcn layer
EDGE_INDEX = torch.tensor([
#Palm          #Thumb        #Index-f   #Middle-f   #Ring-f     #Pinky
[0,5,9,13,17,  0, 1, 2, 3,   5, 6, 7,   9,10,11,    13,14,15,   17,18,19],
[5,9,13,17,0,  1, 2, 3, 4,   6, 7, 8,   10,11,12,   14,15,16,   18,19,20]
], dtype=torch.long, device=DEVICE) #long requested by gcn



#===========================
#Classficator Training
#===========================
BATCH_SIZE = 1 #for inference use BATCH_SIZE = 1
GESTURE_SAMPLE_PATH = r'D:\Arbeit\_Studium (Derzeit)\bsc\Src\Trainingsamples' #Parentfolder of the training-samples for the network
LOSS_FUNCTION = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
NUM_EPOCHS = 35
REL_PORTION_FOR_VALIDATION = 0.15
REL_PORTION_FOR_TESTING = 0.15

#============================
#Unit-testing
#============================
SAMPLE_PICTURE_PATH = r'D:\Arbeit\_Studium (Derzeit)\bsc\Src\Hand_Recognition\Tests\TestImage.jpg'