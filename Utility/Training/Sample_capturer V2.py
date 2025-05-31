import sys
import threading


import mediapipe as mp
import time

import cv2
import torch
from Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, NUM_CHANNELS_LAYER2, \
    KERNEL_SIZE, DROPOUT, DEVICE, STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from Utility.Dataclasses import SpatialFeaturePackage
from Utility.Enums import Gesture
from Utility.Exceptions import UnsuccessfulCaptureException
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.FeatureExtractor import FeatureExtractor, FeaturePackage
from PipelineModules.FrameCapturer import FrameCapturer

ESC = 27

def main() -> None:
        stop_event = threading.Event()

        lm_capturer: FrameCapturer = FrameCapturer(stop_event)
        capture_thread = threading.Thread(target=lm_capturer.run, daemon=True)
        capture_thread.start()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
        out = cv2.VideoWriter('cur_sample.mp4', fourcc, 30.0, (1280, 720))

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        mediapipe = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        t0 = 0
        t1 = 0
        while True:
            t0 = t1
            t1 = time.time()
            frame = lm_capturer.get()
            out.write(frame)
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = mediapipe.process(RGB_frame)
            if mp_result.multi_hand_landmarks:
                for hand_landmarks in mp_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Debug", frame)
            print("fps: " + str(1/(t1-t0)))
            if cv2.waitKey(1) == ESC:
                stop_event.set()
                capture_thread.join()
                break




if __name__ == '__main__':
    main()