import os

import cv2
from torch import torch,nn

from src.Utility.Enums import FrameDropLoggingMode, Gesture

#TODO sort in
FRAME_CAPTURER_QUEUE_SIZE = 3

#Collection of all setting-constants used by the pipeline modules
#===========================
#Debug tools
#===========================
DEBUG = True #when false: disables every debug feature (OVERWRITES: DEBUG_SHOW_IMAGE,DEBUG_SHOW_NUM_FRAMES_DROPPED, DEBUG_PRINT_RESULTS)

DEBUG_PRINT_RESULTS = True
DEBUG_SHOW_IMAGE = True
DEBUG_SHOW_NUM_FRAMES_DROPPED = FrameDropLoggingMode.OFF


#===========================
#FrameCapturer
#===========================
IMAGE_SOURCE = 0 #droid cam: 0 webcam: 1
CAPTURE_BACKEND = cv2.CAP_MSMF #cv2.CAP_DSHOW | cv2.CAP_MSMF | cv2.CAP_ANY
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 30 #not supported by all cameras

SKIP_N_FRAMES = 0 #drop frames inbetween for more consistency or debugging


#===========================
#LmExtractor
#===========================
#Mediapipe
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MP_MODEL_COMPLEXITY = 1

NUM_LANDMARKS = 21 #needed for other components no dynamic assignments possible
NUM_LANDMARK_DIMENSIONS = 3

#For using mediapipe simultaneously on different frames
#The option gives improvements on higher end systems (doesn't make any sense but works on my machine)
USE_CUSTOM_MP_MULTITHREADING = True
EXTRACTOR_NUM_THREADS = 2 #Mainly used by the mediapipe extraction
CLASSIFICATOR_NUM_THREADS = 6
CLASSIFICATOR_NUM_INTEROP_THREADS = 2

#Filtersettings
FILTER_CONSEC_FRAMEDROPS = 3
USE_FILTER = True

#===========================
#Classificator
#===========================
FRAMEWINDOW_LEN = 31
KERNEL_SIZE = 3
DROPOUT = 0.2 #gets applied once after the gcn and inside the temporal blocks of the tcn
DEVICE = 'cuda:0' # cpu | cuda:0 | cuda

GCN_NUM_OUTPUT_CHANNELS = 36
TCN_INPUT_SIZE = NUM_LANDMARKS * GCN_NUM_OUTPUT_CHANNELS + 1 #+1: HAND_DETECTED feature
TCN_CHANNELS = [FRAMEWINDOW_LEN, FRAMEWINDOW_LEN, FRAMEWINDOW_LEN, FRAMEWINDOW_LEN] #receptive field of 31
TCN_NUM_OUTPUT_CLASSES = 7 #Nothing, ScrollUp, ScrollDown, SwipeLeft, SwipeRight, ZoomIn, ZoomOut

#Edge index for the gcn layer
EDGE_INDEX = torch.tensor([
#Palm          #Thumb        #Index-f   #Middle-f   #Ring-f     #Pinky
[0,5,9,13,17,  0, 1, 2, 3,   5, 6, 7,   9,10,11,    13,14,15,   17,18,19],
[5,9,13,17,0,  1, 2, 3, 4,   6, 7, 8,   10,11,12,   14,15,16,   18,19,20]
], dtype=torch.long, device=DEVICE) #long requested by gcn

WEIGHTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "PipelineModules", "Classificator", "trained_weights.pth")

#===========================
#EventHandler
#===========================
SEND_ACROBAT_EVENTS = False
PIPE_NAME = r"\\.\pipe\AcrobatGestureRecognition"

CONFIDENCE_THRESHOLD = 0.99
GESTURE_COOLDOWN_S = 1.2

#===========================
#Classficator Training
#===========================
GESTURE_SAMPLE_PATH =  os.path.join(os.path.dirname(__file__), "..", "trainingsamples")
SLEEP_BETWEEN_SAMPLES_S = 0 #for checking the samples

NUM_EPOCHS = 20 #14,35

#learning
LOSS_FUNCTION = nn.CrossEntropyLoss()
LEARNING_RATE_INIT = 0.0003 #0.0001
LEARNING_RATE_DECAY_FACTOR = 0.65
LEARNING_RATE_STEPS = 2

#datasets
BATCH_SIZE = 3
REL_PORTION_FOR_VALIDATION = 0.05
REL_PORTION_FOR_TESTING = 0.05

#artificial samples
USE_ARTIFICIAL_SAMPLES = True
ARTIFICIAL_SAMPLES_NOISE = 0.0015 #previous 0.0045
ARTIFICIAL_SAMPLES_OFFSET_NOISE = 0.35
ARTIFICIAL_SAMPLES_PER_SAMPLE = 9
USE_TRAINING_WEIGHTS = False
TRAINING_WEIGHTS = [0.5] * 20 + [1.0] * (TCN_INPUT_SIZE - 20)

#============================
#Unit-testing
#============================
SAMPLE_PICTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "Tests", "Training", "TestImage.jpg")