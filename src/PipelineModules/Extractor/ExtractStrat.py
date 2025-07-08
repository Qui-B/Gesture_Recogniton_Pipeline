import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable
import concurrent.futures

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, \
    MIN_TRACKING_CONFIDENCE, DEVICE, EXTRACTOR_NUM_THREADS, USE_CUSTOM_MP_MULTITHREADING, MP_MODEL_COMPLEXITY, \
    USE_FILTER, FILTER_CONSEC_FRAMEDROPS
from src.PipelineModules.Extractor.FrameFilter import FrameFilterFactory
from src.Utility.Dataclasses import FeaturePackage
from src.Utility.Exceptions import NullPointerException
from src.Utility.DebugManager.DebugManager import debug_manager


class ExtractStratFactory:
    @staticmethod
    def get(getFrame: Callable[[], any],
            use_custom_mp_multithreading: bool,
            start_event: [threading.Event] = None,
            stop_event: [threading.Event] = None,
            use_Filter: bool = USE_FILTER):

        if use_custom_mp_multithreading:
            if start_event is None or stop_event is None:
                raise NullPointerException()
            else:
                return ExtractStratFactory.ExtractStratMPMultiThreading(getFrame, start_event, stop_event, use_Filter)
        else:
            return ExtractStratFactory.ExtractStratMPSingleThreading(getFrame, use_Filter)

    class ExtractStratBase(ABC):
        def __init__(self,
                     getFrame: Callable[[], any],
                     use_filter: bool = USE_FILTER):

            self.getFrame = getFrame
            self.mp = mp.solutions.hands.Hands(
                static_image_mode=STATIC_IMAGE_MODE,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                model_complexity=MP_MODEL_COMPLEXITY
            )
            self.filter = FrameFilterFactory.get(use_filter)


        def extract(self, frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB colorscheme needed for landMarc extraction
            mp_result = self.mp.process(rgb_frame)
            landmark_coordinates = np.zeros((21, 3))
            hand_detected = False

            if mp_result.multi_hand_landmarks:
                landmarks = mp_result.multi_hand_landmarks[0].landmark
                for index, landmark in enumerate(landmarks):
                    landmark_coordinates[index] = [landmark.x, landmark.y, landmark.z]
                hand_detected = True
                debug_manager.render(frame, mp_result.multi_hand_landmarks[0])

            feature_package = FeaturePackage(
                torch.tensor(landmark_coordinates, dtype=torch.float32, device=DEVICE),
                hand_detected
            )
            return feature_package

        @abstractmethod
        def getNext(self):
            pass

    class ExtractStratMPSingleThreading(ExtractStratBase):
        def __init__(self,
                     getFrame: Callable,
                     use_filter: bool = USE_FILTER):
            super().__init__(getFrame, use_filter)

        def getNext(self):
            while True:
                rgb_frame = self.getFrame()
                feature_package = self.extract(rgb_frame)
                if self.filter.validate(feature_package.hand_detected):
                    break
            debug_manager.show(rgb_frame)
            return feature_package

    class ExtractStratMPMultiThreading(ExtractStratBase):
        def __init__(self, getFrame, start_event: [threading.Event], stop_event: [threading.Event], use_filter=USE_FILTER):
            super().__init__(getFrame, use_filter)
            self.stop_event = stop_event
            self.start_event = start_event
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=EXTRACTOR_NUM_THREADS)
            self.feature_package_futures = queue.Queue(maxsize=6)

        def start(self):
            extractor_thread = threading.Thread(target=self.run)
            extractor_thread.start()

        def cancelFutures(self):
            while not self.feature_package_futures.empty():
                self.feature_package_futures.get_nowait().cancel()

        def run(self):
            self.start_event.wait()
            while not self.stop_event.is_set():
                future = self.executor.submit(self.runThread)
                try:
                    self.feature_package_futures.put(future, timeout=1)
                except queue.Full:
                    debug_manager.log_framedrop("FeatureExtractor.run()")
            self.cancelFutures()
            self.executor.shutdown(wait=True)
            print("Extractor-module stopped")

        def runThread(self):
            try:
                frame = self.getFrame() #getFrame inside to thread for low latency
                feature_package = self.extract(frame)
                return feature_package, frame
            except queue.Empty:
               debug_manager.log_framedrop("FeatureExtractor.runThread() - getFrame")

        def getNext(self):
            while True:
                future = self.feature_package_futures.get(block=True,timeout=10)  # throws queue.Empty, handled in App to avoid error
                feature_package, frame = future.result()

                if self.filter.validate(feature_package.hand_detected):
                    break
            debug_manager.show(frame)
            return feature_package