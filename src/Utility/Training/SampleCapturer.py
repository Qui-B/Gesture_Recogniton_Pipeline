import threading
import cv2
import mediapipe as mp

from src.Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.PipelineModules.FrameCapturer import FrameCapturer

ESC = 27
def main() -> None:
        stop_event = threading.Event()

        frame_capturer: FrameCapturer = FrameCapturer(stop_event)
        fps = frame_capturer.measure_camera_fps(400)
        print("FPS: " + str(fps))

        actual_width = frame_capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = frame_capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"🔍 Camera resolution: {int(actual_width)}x{int(actual_height)}")

        capture_thread = threading.Thread(target=frame_capturer.run, daemon=True)
        capture_thread.start()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
        out = cv2.VideoWriter('cur_sample.mp4', fourcc, 30.0, (640, 360))

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        mediapipe = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        while True:
            frame = frame_capturer.get()
            out.write(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = mediapipe.process(rgb_frame)
            if mp_result.multi_hand_landmarks:
                for hand_landmarks in mp_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Debug", frame)
            if cv2.waitKey(1) == ESC:
                stop_event.set()
                capture_thread.join()
                break




if __name__ == '__main__':
    main()