
from collections import deque
import numpy as np
import torch

from DebugUtil import printLandMarkCoordinate, printTensor
#from Hand_Recognition.DebugUtil import printAverageDirectionChange, printLandMarkCoordinate
from Settings import WINDOW_LENGTH, FEATURE_VECTOR_LENGTH


class WindowManager:
    """
    stores the most recent landmark vectors in a deque which serves as a input for the classification tcp
    """
    def __init__(self) -> None:
        self.landmark_vector_deque = deque(maxlen=WINDOW_LENGTH)
        self.lastFrame = np.zeros(64)

    def update(self, landmark_vector):
        """
        For the input vector: calculates the relative landmark vector.
        Then converts it to a torch.tensor and appends it to the end of the image-deque.

        Args:
            landmark_vector: length 64 feature vector. (0-62: landmark coordinates; 63: hand present indication)
        """
        self.landmark_vector_deque.append(landmark_vector)









