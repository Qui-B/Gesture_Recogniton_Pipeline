import queue
from typing import Callable
import concurrent.futures
import cv2
import mediapipe as mp
import numpy as np
import torch

from ..Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, \
    MIN_TRACKING_CONFIDENCE, DEVICE, EXTRACTOR_NUM_THREADS
from ..Utility.Dataclasses import FeaturePackage


class FeatureExtractor:
    """
    Extracts landmark-features from an image and captures them in a FeaturePackage.
    Spatial features get extracted duríng the classification process as they also have to be trained in combination with the later tcn layer.
    """

    def __init__(self, stop_event):
        self.lastFrame = None
        self.stop = stop_event
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=EXTRACTOR_NUM_THREADS)  # thread pool
        self.feature_package_futures = queue.Queue(maxsize=6)
        #mediapipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=1
        )

        self.queue_full_failures = 0  # No custom failure handle because of potential performance slowdown
        self.frame_queue_timeout_failure = 0

    def run(self, getRGBFrame: Callable[[], np.array]):
        while not self.stop.is_set():
            try:
                frame = getRGBFrame()  # blocks until frame is there
                future = self.executor.submit(self.extract, frame)
                self.feature_package_futures.put((future,frame))
            except Exception as e:
                if isinstance(e, queue.Empty): #case getRGBFrame reached timeout
                    self.frame_queue_timeout_failure += 1
                elif isinstance(e, queue.Full):
                    self.queue_full_failures += 1

        self.executor.shutdown(wait=True)

    def getNext(self):
        future, frame = self.feature_package_futures.get(block=True)
        feature_package = future.result()
        cv2.imshow("debug", frame) #DEBUG: For checking future starving
        cv2.waitKey(1)
        return feature_package

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
        self.lastFrame = landmark_vector #TODO put in loop and check
        return relative_landmark_vector

    def sharpen_frame(self,frame): #MAYBE DELETE
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)




    def extract(self, rgb_frame):
        """
        Extracts the hand landmarks from an image. Converts the coordinates to the differences from the previous frame,
        and returns them as a FeaturePackage.

        Args:
            rgb_frame: base image-frame from which the features get extracted

        Returns:
            feature-package (object): Contains a (21x3) tensor of landmarks and a flag indicating if a hand was detected.
        """
        mp_result = self.mp.process(rgb_frame)

        landmark_coordinates = np.zeros((21, 3))
        hand_detected = False

        if mp_result.multi_hand_landmarks:
            landmarks = mp_result.multi_hand_landmarks[0].landmark
            for index, landmark in enumerate(landmarks):
                landmark_coordinates[index] = [landmark.x, landmark.y, landmark.z]
            for hand_landmarks in mp_result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            hand_detected = True

        relative_landmark_vector = self.calcRelativeVector(landmark_coordinates)
        feature_package = FeaturePackage(
            torch.tensor(relative_landmark_vector, dtype=torch.float32, device=DEVICE),
            hand_detected
        )
        return feature_package

