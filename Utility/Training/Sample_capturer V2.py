import cv2
import mediapipe as mp
import time

from Config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS, STATIC_IMAGE_MODE

ESC = 27


def main() -> None:
    cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_FPS, 30)



    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(
        static_image_mode=STATIC_IMAGE_MODE,  # For real-time video
        max_num_hands=MAX_NUM_HANDS,  # Detect one or both hands
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
    out = cv2.VideoWriter('cur_sample.mp4', fourcc, 30.0, (1280, 720))
    timestamp = 0
    successfulRead, image = cap.read()
    while True:
        print("fps: " + str(1/(time.time()-timestamp)))
        timestamp = time.time()
        if cv2.waitKey(1) == ESC:
            break

        successfulRead, image = cap.read()
        if successfulRead:
            out.write(image)
            RGB_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            normal_mpResult = hand.process(RGB_frame)

            out.write(RGB_frame)

            if normal_mpResult.multi_hand_landmarks:
                for hand_landmarks in normal_mpResult.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(RGB_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            cv2.imshow("Debug", RGB_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
