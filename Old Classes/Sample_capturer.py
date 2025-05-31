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



def main() -> None:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(
        static_image_mode=False,  # For real-time video
        max_num_hands=1,  # Detect one or both hands
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
    out = cv2.VideoWriter('cur_sample.mp4', fourcc, 30.0, (1280, 720))

    stop_event = threading.Event()
    lm_capturer = FrameCapturer(stop_event)
    capture_thread = threading.Thread(target=lm_capturer.run)
    capture_thread.start()

    while not stop_event.is_set():
        if cv2.waitKey(1) == ESC:
            break

        frame = lm_capturer.get()
        out.write(frame)
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = hand.process(RGB_frame)
        #out.write(RGB_frame)
        if mp_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(RGB_frame, mp_result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Debug", RGB_frame)

    capture_thread.join()


if __name__ == '__main__':
    main()
