"""
import numpy as np

from DebugUtil import printNPArray



# Set random seed for reproducibility
np.random.seed(42)

# Random landmark vectors
rel_vector_1 = np.random.randint(0, 9, size=(21, 3))
rel_vector_2 = np.random.randint(0, 9, size=(21, 3))
rel_vector_3 = np.random.randint(0, 9, size=(21, 3))

# Subtract rel_vector_2 from rel_vector_1 and rel_vector_2 from rel_vector_3
result_1 = rel_vector_2 - rel_vector_1
result_2 = rel_vector_3 -rel_vector_2


# Print the resulting vectors
print("Result of rel_vector_1 - rel_vector_2:")
print(printNPArray(result_1))

print("\nResult of rel_vector_2 - rel_vector_3:")
print(printNPArray(result_2))
"""
from Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, NUM_CHANNELS_LAYER2, \
    KERNEL_SIZE, DROPOUT
from PipelineModules.Classificator.GraphTCN import GraphTcn

"""
import timeit
from PipelineModules.DataClasses import FeaturePackage, SpatialFeaturePackage
from collections import namedtuple

iterations = 10000000
lm_data_np = [0] * 63  # 21 landmarks * 3 coordinates
hand_flag = True


def createPlain():
    return (lm_data_np, hand_flag)

def createNamed():
    return FeaturePackage(lm_data_np, hand_flag)


time_plain_tuple = timeit.timeit('createPlain',
                                 setup='from __main__ import createPlain',
                                 number=iterations)

time_named_tuple = timeit.timeit('createNamed',
                                setup='from __main__ import createNamed',
                                number=iterations)
avg_plain_tuple = (time_plain_tuple / iterations) * 1000
avg_named_tuple = (time_named_tuple / iterations) * 1000
print("Named_Tuple:   " + str(avg_plain_tuple) + "ms")
print("Unnamed_Tuple: " + str(avg_named_tuple) + "ms")
"""
"""
import cv2

cap = cv2.VideoCapture(0)
print("Webcam fps: " +str(cap.get(cv2.CAP_PROP_FPS)))
"""
"""
import torch
print(torch.__version__)                  # e.g., '2.2.0+cpu' or '2.2.0+cu118'
print(torch.version.cuda)                # None if CUDA is not included
print(torch.cuda.is_available())         # Should be True
"""
"""
import torch

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now create the model directly on the correct device
classificator = GraphTcn(
    input_size=INPUT_SIZE,
    output_size=NUM_OUTPUT_CLASSES,
    gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
    num_channels_layer1=NUM_CHANNELS_LAYER1,
    num_channels_layer2=NUM_CHANNELS_LAYER2,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT
)
classificator.to(device)
# Check if the model is on the correct device
print("Model is on device:", next(classificator.parameters()).device)
"""