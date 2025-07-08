import cv2
import mediapipe as mp
import numpy
import torch
from mediapipe.framework.formats import landmark_pb2

from src.Utility.Dataclasses import FeaturePackage
from src.Utility.DebugManager.DebugManager import debug_manager
from src.Config import MIN_DETECTION_CONFIDENCE, MAX_NUM_HANDS, MIN_TRACKING_CONFIDENCE, \
    USE_ARTIFICIAL_SAMPLES


def printCoord(coord_num, coords):
    print(f"coord {coord_num}: ({coords[coord_num].x:.4f}/{coords[coord_num].y:.4f}/{coords[coord_num].z:.4f})")


def tensor_to_mediapipe_landmarks(tensor_landmarks: torch.Tensor):
    mp_landmarks = landmark_pb2.NormalizedLandmarkList()
    coords = tensor_landmarks.cpu().numpy()

    for x, y, z in coords:
        landmark = mp_landmarks.landmark.add()
        landmark.x = x
        landmark.y = y
        landmark.z = z

    return mp_landmarks

video_source = r'E:\Test\sample_filtered.avi'
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

mediapipe = mp.solutions.hands.Hands(
        static_image_mode=True, #needed for determinism lm-detection
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

while True:
    ret, frame = cap.read()


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not ret:
        print("End of video or cannot read the frame.")
        break

    mp_result = mediapipe.process(rgb_frame)

    if mp_result.multi_hand_landmarks:
        # weird but works
        hand_landmarks = mp_result.multi_hand_landmarks[0].landmark
        coords_np = numpy.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=numpy.float32)
        print(f"shape: {coords_np.shape}, sum: {coords_np.sum()}")
        feature_package = FeaturePackage(torch.from_numpy(coords_np), True)

        if USE_ARTIFICIAL_SAMPLES:
            feature_package = feature_package.applyNoise()

        debug_manager.render(frame, tensor_to_mediapipe_landmarks(feature_package.lm_coordinates))
        #--------
        printCoord(8,mp_result.multi_hand_landmarks[0].landmark)

    cv2.imshow('Video Frame', frame)

    # Wait for 1 ms and check if 'q' is pressed to quit
    key_pressed = cv2.waitKey(0)
    if key_pressed & 0xFF == ord('q'):
        break

# Release the video object and close the window
cap.release()
cv2.destroyAllWindows()

