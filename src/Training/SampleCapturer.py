import os
import threading
import cv2
import keyboard
import mediapipe as mp

from src.Config import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.PipelineModules.Extractor import FrameFilter
from src.PipelineModules.Extractor.FrameFilter import FrameFilterFactory
from src.PipelineModules.FrameCapturer import FrameCapturer
from src.Utility.DebugManager.DebugManager import debug_manager

ESC = 27
VIDEONAME = "sample.avi"

#TODO make a generic solution that takles a consumer as lambda so filter and capturre can be done with one method


def filterVid():
    frame_filter = FrameFilterFactory.FrameFilter()

    current_dir = os.getcwd()
    video_path = os.path.join(current_dir, VIDEONAME)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use 'avc1' or 'H264' if needed
    filtered_out = out = cv2.VideoWriter("sample_filtered.avi", fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    mediapipe = mp.solutions.hands.Hands(
            static_image_mode=True, #needed for detection determinism
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = mediapipe.process(RGB_frame)
        if frame_filter.validate(mp_result.multi_hand_landmarks):
            filtered_out.write(frame)

    cap.release()
    out.release()

def main() -> None:
        start_event = threading.Event()
        stop_event = threading.Event()

        frame_capturer: FrameCapturer = FrameCapturer(start_event, stop_event)
        frame_filter: FrameFilter = FrameFilterFactory.get(True)
        fps = frame_capturer.measure_camera_fps(300)
        print(f"Capture-module initialization succeeded [FPS: {fps}, "
              f"Resolution: {int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}]")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use 'avc1' or 'H264' if needed
        out = cv2.VideoWriter(VIDEONAME, fourcc, int(fps),
                              (int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(frame_capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print("Videowriter initialized")

        mediapipe = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=0 #to avoid dropped frames
        )
        frame_capturer.spawn_thread()
        start_event.set()
        while not stop_event.is_set():
            try:
                frame = frame_capturer.get()
                RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = mediapipe.process(RGB_frame)


                out.write(frame)
                if mp_result.multi_hand_landmarks: #multi hands can still be none
                    debug_manager.render(frame, mp_result.multi_hand_landmarks[0])
                debug_manager.show(frame)
                debug_manager.print_framedrops()

            except Exception:
                debug_manager.log_framedrop("SampleCapturer.main()")
            debug_manager.print_framedrops()
            if keyboard.is_pressed('esc'):
                print("Vid capture stopped")
                out.release()
                stop_event.set()
                debug_manager.close()
        filterVid()




if __name__ == '__main__':
    main()