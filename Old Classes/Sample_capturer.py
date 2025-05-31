import sys
import threading
import time
from typing import Callable
import cv2
import mediapipe as mp

import numpy as np

from Config import FPS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SOURCE, STATIC_IMAGE_MODE, MAX_NUM_HANDS, \
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from PipelineModules.FrameCapturer import FrameCapturer
from Utility.Dataclasses import FeaturePackage

ESC = 27



class VideoCreator:
    def __init__(self, stop_event, getFrame:  Callable[[], np.ndarray]):
        self.getFrame  = getFrame
        self.stop_event = stop_event
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )



    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
        out = cv2.VideoWriter('cur_sample.mp4', fourcc, 30.0, (1280, 720))

        count = 0
        frameTime = 0

        while not self.stop_event.is_set():
            frame = self.getFrame()
            t0 = time.time()
            count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = self.mp.process(frame_rgb)
            if mp_result.multi_hand_landmarks:
                for hand_landmarks in mp_result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp.HAND_CONNECTIONS)
            cv2.imshow("debug window", frame)
            out.write(frame)
            if cv2.waitKey(1) == ESC:
                self.stop_event.set()
            if count == 100:
                sys.stdout.write(f"fps: {frameTime/100}")
                sys.stdout.write('\033[H')
                sys.stdout.flush()
                frameTime = 0
                count = 0
            else:
                frameTime += 1 / (time.time() - t0)

        cv2.destroyAllWindows()



def main() -> None:
    stop_event = threading.Event()

    lm_capturer = FrameCapturer(stop_event)
    capture_thread = threading.Thread(target=lm_capturer.run)

    recorder = VideoCreator(stop_event, lm_capturer.get)
    recorder_thread = threading.Thread(target=recorder.run)

    capture_thread.start()
    recorder_thread.start()

    capture_thread.join()
    recorder_thread.join()


if __name__ == '__main__':
    main()
