import random

import numpy as np
import torch
from typing import NamedTuple

from src.Config import ARTIFICIAL_SAMPLES_NOISE


#Various dataclasses for transferring data between pipeline modules

class FeaturePackage(NamedTuple):
    """
       Dataclass used to store the features, which get extracted by the ExtractorBase. Gets processed by the Graph-TCN model.

       Fields:
           lm_coordinates (np.array 21*3): Holds the coordinates (x,y,z) for each landmark given from mediapipe
           hand_detected (float): indicates if the hand was detected
    """
    lm_coordinates: torch.tensor
    hand_detected: bool

    def applyNoise(self, standard_deviation:float = ARTIFICIAL_SAMPLES_NOISE):
        lm_coordinates_clone = self.lm_coordinates.clone()
        noise = torch.normal(
            mean=0.0,
            std=standard_deviation,
            size=self.lm_coordinates.shape,
            device=self.lm_coordinates.device
        )
        lm_coordinates_clone += noise

        noisy_feature_package = FeaturePackage(lm_coordinates_clone, self.hand_detected)
        return noisy_feature_package

class SpatialFeaturePackage(NamedTuple):
    """
           Dataclass used to store the extracted spatial features from the first layer of the Graph-TCN, which get extracted by the ExtractorBase. Gets processed by the Graph-Tcn model.

           Fields:
               spatial_lm_t (torch.Tensor 21*12): Holds 12 spatial relationships for every landmark
               hand_detected (float): indicates if the hand was detected
        """
    spatial_lm_t: torch.Tensor
    hand_detected: bool


class TrainingSample(NamedTuple):
    feature_packages: list[FeaturePackage]
    label: torch.Tensor

    def applyNoise(self, standard_deviation:float = ARTIFICIAL_SAMPLES_NOISE):
        noisy_feature_packages = []
        for feature_package in self.feature_packages:
            noisy_feature_package = feature_package.applyNoise(standard_deviation)
            noisy_feature_packages.append(noisy_feature_package)


        noisy_training_sample = TrainingSample(noisy_feature_packages, self.label)
        return noisy_training_sample

#No data container for landmark coordinates to improve performance