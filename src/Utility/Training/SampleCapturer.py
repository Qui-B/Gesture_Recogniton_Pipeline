import threading
import cv2
import keyboard
import mediapipe as mp

from src.Config import STATIC_IMAGE_MODE, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, \
    DEBUG_SHOW_NUM_FRAMES_DROPPED
from src.PipelineModules.FrameCapturer import FrameCapturer
from src.Utility.DebugManager.DebugManager import debug_manager
from src.Utility.Enums import FrameDropLoggingMode

ESC = 27
def main() -> None:
        start_event = threading.Event()
        stop_event = threading.Event()

        frame_capturer: FrameCapturer = FrameCapturer(start_event, stop_event)
        fps = frame_capturer.measure_camera_fps(600)
        print(f"Capture-module initialization succeeded [FPS: {fps}, "
              f"Resolution: {int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}]")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
        out = cv2.VideoWriter('sample_droid_cam_dcap.mp4', fourcc, int(fps),
                              (int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print("Videowriter initialized")

        mediapipe = mp.solutions.hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        frame_capturer.spawn_thread()
        start_event.set()
        while not stop_event.is_set():
            try:
                frame = frame_capturer.get()
                out.write(frame)
                RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = mediapipe.process(RGB_frame)

                if mp_result.multi_hand_landmarks:
                    debug_manager.render(frame, mp_result.multi_hand_landmarks[0])
                debug_manager.show(frame)
            except Exception:
                debug_manager.log_framedrop("SampleCapturer.main()")
            debug_manager.print_framedrops()
            if keyboard.is_pressed('esc'):
                print("ESC pressed shutting down...")
                out.release()
                stop_event.set()
                debug_manager.close()



if __name__ == '__main__':
    main()