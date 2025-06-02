from torch import torch,nn

#Collection of all setting-constants used by the pipeline modules

#===========================
#FrameCapturer
#===========================
IMAGE_SOURCE = 1 #droid cam: 0 webcam: 1
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
SKIP_N_FRAMES = 0 #drop frames inbetween for more consistency

#===========================
#LmExtractor
#===========================
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_LANDMARKS = 21
NUM_LANDMARK_DIMENSIONS = 3

#Used parallelizing mediapipe on different frames
USE_CUSTOM_MP_MULTITHREADING = True #Gives improvements on higher end systems (doesn't make any sense)
EXTRACTOR_NUM_THREADS = 2 #Mainly used by the mediapipe extraction
CLASSIFICATOR_NUM_THREADS = 8
CLASSIFICATOR_NUM_INTEROP_THREADS = 2

#===========================
#Classificator
#===========================
FRAMEWINDOW_LEN = 31
KERNEL_SIZE = 3
DROPOUT = 0.1 #TESTING
DEVICE = 'cuda:0' # or cpu

GCN_NUM_OUTPUT_CHANNELS = 36
INPUT_SIZE = NUM_LANDMARKS * GCN_NUM_OUTPUT_CHANNELS + 1 #+1: HAND_DETECTED feature | NUM_LANDMARKS also get used for input_size
NUM_CHANNELS_LAYER1 = [FRAMEWINDOW_LEN, FRAMEWINDOW_LEN, FRAMEWINDOW_LEN, FRAMEWINDOW_LEN] #receptive field of 31
NUM_OUTPUT_CLASSES = 7 #Nothing, ScrollUp, ScrollDown, SwipeLeft, SwipeRight, ZoomIn, ZoomOut

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