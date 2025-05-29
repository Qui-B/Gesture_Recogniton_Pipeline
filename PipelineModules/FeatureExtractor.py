import cv2
import mediapipe as mp
import numpy as np
import torch

from Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, \
    MIN_TRACKING_CONFIDENCE, DEVICE
from Utility.Dataclasses import FeaturePackage


class FeatureExtractor:
    """
    Extracts landmark-features from an image and captures them in a FeaturePackage.
    Spatial features get extracted duríng the classification process as they also have to be trained in combination with the later tcn layer.
    """
    def __init__(self):
        self.lastFrame = None

        #mediapipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def calcRelativeVector(self, landmark_vector):
        """
            Transforms the landmark coordinates from a landmark_vector, so the coordinates represent the difference to the previous landmark_vector.
            The first frame gets transformed to all zeros.

            Args:
                array[float]: feature vector (21 landmarks * 3 coordinates) from which the relative vector is computed.

            Returns:
                array[float]: feature vector (21 * 3) Where every value is the difference to the feature vector of the last frame.
        """
        relative_landmark_vector = np.zeros((21,3))
        if self.lastFrame is not None:
            relative_landmark_vector[:,:] = landmark_vector[:,:] - self.lastFrame[:,:]
        self.lastFrame = landmark_vector
        return relative_landmark_vector

    def sharpen_frame(self,frame):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)


    def extract(self, RGB_frame):
        """
        Extracts the hand landmarks from an image. Afterward converts the coordinates to the distances from the previous image and
        returns them as a feature-package.

        Args:
            RGB_frame: base image-frame from which the features get extracted

        Returns:
            feature-package (object): Contains an array (21*3) for storing the landmarks and an extra field indicating if a hand got detected.
        """
        landmark_coordinates = np.zeros((21,3))
        hand_detected = False

        sharpened_frame = self.sharpen_frame(RGB_frame)
        mp_result = self.mp.process(sharpened_frame)

        if mp_result.multi_hand_landmarks: #case: hand detected
            landmarks = mp_result.multi_hand_landmarks[0].landmark
            for index, landmark in enumerate(landmarks):
                landmark_coordinates[index] = [landmark.x, landmark.y, landmark.z]
            self.mp_drawing.draw_landmarks(RGB_frame, mp_result.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            hand_detected = True

        cv2.imshow("Debug Frame",RGB_frame)
        cv2.waitKey(30)

        relative_landmark_vector = self.calcRelativeVector(landmark_coordinates)
        feature_package = FeaturePackage(torch.tensor(relative_landmark_vector, dtype=torch.float32, device=DEVICE), hand_detected)
        return feature_package

