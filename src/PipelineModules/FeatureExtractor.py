import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable
import concurrent.futures
import cv2
import mediapipe as mp
import numpy as np
import torch

from ..Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, \
    MIN_TRACKING_CONFIDENCE, DEVICE, EXTRACTOR_NUM_THREADS, DEBUG, USE_CUSTOM_MP_MULTITHREADING, MP_MODEL_COMPLEXITY
from ..Utility.Dataclasses import FeaturePackage
from ..Utility.Exceptions import NullPointerException

"""
Extracts landmark-features from an image and captures them in a FeaturePackage.
Spatial features get extracted duríng the classification process as they also have to be trained in combination with the later tcn layer.
"""

class FeatureExtractor():
    def __init__(self,
                 getFrame: Callable[[], any],
                 start_event: threading.Event = None,
                 stop_event: threading.Event = None,
                 use_custom_mp_multithreading: bool = USE_CUSTOM_MP_MULTITHREADING,
                 debug: bool = DEBUG):
        self.debugManager = DebugManagerFactory.get(debug)
        self.extractStrat = ExtractStratFactory.get(getFrame, self.debugManager, use_custom_mp_multithreading, start_event, stop_event)

    def start(self):
        self.extractStrat.start()

    def get(self):
        return self.extractStrat.getNext()

class DebugManagerFactory:
    @staticmethod
    def get(debug: bool):
        return DebugManagerFactory.DebugManager() if debug else DebugManagerFactory.NoDebugManager()

    class DebugManagerBase(ABC):
        @abstractmethod
        def render(self, frame, landmarks):
            pass

        @abstractmethod
        def draw(self, frame):
            pass

        @abstractmethod
        def close(self):
            pass

    class DebugManager(DebugManagerBase):
        def __init__(self):
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

        def render(self, frame, landmarks):
            self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        def draw(self, frame):
            cv2.imshow("Debug", frame)
            cv2.waitKey(1)

        def close(self):
            cv2.destroyAllWindows()

    class NoDebugManager(DebugManagerBase):
        def render(self, frame, landmarks ):
            pass

        def draw(self, frame):
            pass

        def close(self):
            pass


class ExtractStratFactory():
    @staticmethod
    def get(getFrame: Callable[[], any],
            debugManager: DebugManagerFactory.DebugManagerBase,
            use_custom_mp_multithreading: bool,
            start_event: [threading.Event] = None,
            stop_event: [threading.Event] = None):

        if use_custom_mp_multithreading:
            if start_event is None or stop_event is None:
                raise NullPointerException()
            else:
                return ExtractStratFactory.ExtractStratMPMultiThreading(getFrame, debugManager, start_event, stop_event)
        else:
            return ExtractStratFactory.ExtractStratMPSingleThreading(getFrame, debugManager)

    class ExtractStratBase(ABC):
        def __init__(self, getFrame: Callable[[], any], debugManager):
            self.getFrame = getFrame
            self.debugManager = debugManager
            self.lastFrame = None
            self.mp = mp.solutions.hands.Hands(
                static_image_mode=STATIC_IMAGE_MODE,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                model_complexity=MP_MODEL_COMPLEXITY
            )

        def calcRelativeVector(self, landmark_vector):
            relative_landmark_vector = np.zeros((21, 3))
            if self.lastFrame is not None:
                relative_landmark_vector[:, :] = landmark_vector[:, :] - self.lastFrame[:, :]
            self.lastFrame = landmark_vector
            return relative_landmark_vector

        def extract(self, rgb_frame):
            mp_result = self.mp.process(rgb_frame)
            landmark_coordinates = np.zeros((21, 3))
            hand_detected = False

            if mp_result.multi_hand_landmarks:
                landmarks = mp_result.multi_hand_landmarks[0].landmark
                for index, landmark in enumerate(landmarks):
                    landmark_coordinates[index] = [landmark.x, landmark.y, landmark.z]
                hand_detected = True
                self.debugManager.render(rgb_frame, mp_result.multi_hand_landmarks[0])

            relative_landmark_vector = self.calcRelativeVector(landmark_coordinates)
            feature_package = FeaturePackage(
                torch.tensor(relative_landmark_vector, dtype=torch.float32, device=DEVICE),
                hand_detected
            )
            return feature_package

        @abstractmethod
        def getNext(self):
            pass

    class ExtractStratMPSingleThreading(ExtractStratBase):
        def __init__(self, getFrame, debugManager):
            super().__init__(getFrame, debugManager)

        def getNext(self):
            rgb_frame = self.getFrame()
            featurePackage = self.extract(rgb_frame)
            self.debugManager.draw(rgb_frame)
            return featurePackage

    class ExtractStratMPMultiThreading(ExtractStratBase):
        def __init__(self, getFrame, debugManager, start_event: [threading.Event], stop_event: [threading.Event]):
            super().__init__(getFrame, debugManager)
            self.stop_event = stop_event
            self.start_event = start_event
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=EXTRACTOR_NUM_THREADS)
            self.feature_package_futures = queue.Queue(maxsize=6)

        def start(self):
            extractorThread = threading.Thread(target=self.run)
            extractorThread.start()

        def run(self):
            self.start_event.wait()
            with self.executor:
                while not self.stop_event.is_set():
                    try:
                        future = self.executor.submit(self.runThread)
                        self.feature_package_futures.put(future)
                    except Exception as e: #TODO
                        print("[run] Exception:", repr(e))
            self.executor.shutdown(wait=True)
            self.debugManager.close()

        def runThread(self):
            frame = self.getFrame() #getFrame inside to thread for low latency
            feature_package = self.extract(frame)
            return feature_package, frame


        def getNext(self):
            future = self.feature_package_futures.get(block=True, timeout=10)
            feature_package, frame = future.result()
            self.debugManager.draw(frame)
            return feature_package
