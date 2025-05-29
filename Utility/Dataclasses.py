from typing import NamedTuple

import numpy as np
import torch


class FeaturePackage(NamedTuple):
    """
       Dataclass used to store the features, which get extracted by the FeatureExtractor. Gets processed by the Graph-TCN model.

       Fields:
           lm_coordinates (np.array 21*3): Holds the coordinates (x,y,z) for each landmark given from mediapipe
           hand_detected (float): indicates if the hand was detected
    """
    lm_coordinates: np.array
    hand_detected: bool

class SpatialFeaturePackage(NamedTuple):
    """
           Dataclass used to store the extracted spatial features from the first layer of the Graph-TCN, which get extracted by the FeatureExtractor. Gets processed by the Graph-Tcn model.

           Fields:
               spatial_lm_t (torch.Tensor 21*12): Holds 12 spatial relationships for every landmark
               hand_detected (float): indicates if the hand was detected
        """
    spatial_lm_t: torch.Tensor
    hand_detected: bool

#Dataclass Trainings_Sample is in GraphTCNTrainer because it's never needed somewhere else