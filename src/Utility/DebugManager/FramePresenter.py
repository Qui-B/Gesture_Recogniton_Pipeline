from abc import ABC, abstractmethod

import cv2
import mediapipe as mp


class DebugFramePresenterFactory:
    @staticmethod
    def get(debug: bool):
        return DebugFramePresenterFactory.DebugFramePresenter() if debug else DebugFramePresenterFactory.DebugNoFramePresenter()

    class DebugFramePresenterBase(ABC):
        @abstractmethod
        def render(self, frame, landmarks):
            pass

        @abstractmethod
        def draw(self, frame):
            pass

        @abstractmethod
        def close(self):
            pass

    class DebugFramePresenter(DebugFramePresenterBase):
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

    class DebugNoFramePresenter(DebugFramePresenterBase):
        def render(self, frame, landmarks ):
            pass

        def draw(self, frame):
            pass

        def close(self):
            pass