import torch

#Collection of all setting-parameters used by the pipeline modules

#LmCapturer
IMAGE_SOURCE = 'http://10.0.0.170:4747/video'
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 600
FPS = 0.2 #for debugging

#LmExtractor
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FEATURE_VECTOR_LENGTH = 21

#WindowManager
WINDOW_LENGTH = 32

#Classificators
#FEATURE_VECTOR_LENGTH and WINDOW_LENGTH also get used for input_size
KERNEL_SIZE = 3
NUM_OUTPUT_CLASSES = 7 #scroll up/down, swipe left/right, zoom in/out, nothing

#Channels for the tcn layers
NUM_CHANNELS_LAYER1 = [32,32,32]
NUM_CHANNELS_LAYER2 = [16,16,16]

#Edge index for the gcn layer
EDGE_INDEX = torch.tensor([
    #Palm          #Thumb        #Index-f   #Middle-f   #Ring-f     #Pinky
    [0,5,9,13,17,  0, 1, 2, 3,   5, 6, 7,   9,10,11,    13,14,15,   17,18,19],
    [5,9,13,17,0,  1, 2, 3, 4,   6, 7, 8,   10,11,12,   14,15,16,   18,19,20]
], dtype=torch.long) #long requested by gcn

#Used for the training of the Classificators
BATCH_SIZE = 1
GESTURE_SAMPLE_PATH = r'D:\Arbeit\_Studium (Derzeit)\bsc\Src\Trainingsamples' #Parentfolder for the samples used to train the network
