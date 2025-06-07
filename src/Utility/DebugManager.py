import string
import threading
from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

from src.Config import DEBUG, DEBUG_SHOW_IMAGE, DEBUG_SHOW_NUM_FRAMES_DROPPED
from src.Utility.Enums import FrameDropLoggingMode

class DebugManagerBase(ABC):
    @abstractmethod
    def render(self, frame, landmarks):
            pass

    @abstractmethod
    def show(self, frame):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def log_framedrop(self, param):
        pass

    @abstractmethod
    def print_framedrops(self):
        pass

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


class DebugFrameDropCounterFactory(ABC):
    @staticmethod
    def get(logging_mode : FrameDropLoggingMode):
        if logging_mode == FrameDropLoggingMode.OFF:
            return DebugFrameDropCounterFactory.DebugNoFrameDropCounter()

        elif logging_mode == FrameDropLoggingMode.Fast:
            return DebugFrameDropCounterFactory.DebugFrameDropCounterNoneBlocking()

        else:
            return DebugFrameDropCounterFactory.DebugFrameDropCounterBlocking()

    class DebugFrameDropCounterBase(ABC):
        @abstractmethod
        def log_framedrop(self, locationName: string):
            pass

        @abstractmethod
        def printAll(self):
            pass

    class DebugNoFrameDropCounter(DebugFrameDropCounterBase):
        def log_framedrop(self, location: string):
            pass

        def printAll(self):
            pass

    class DebugFrameDropCounter(DebugFrameDropCounterBase):
         @abstractmethod
         def __init__(self):
             self.log = dict()
             self.lock = threading.Lock()

         @abstractmethod
         def log_framedrop(self, location: string):
             pass

         def printAll(self):
             for location, val in self.log.items():
                 print(f"{location}: {val} dropped frames")

    class DebugFrameDropCounterBlocking(DebugFrameDropCounter):
         def __init__(self):
             self.log = dict()
             self.lock = threading.Lock()

         def log_framedrop(self, location: string):
             with self.lock:
                self.log[location] = self.log.get(location, 0) + 1

    class DebugFrameDropCounterNoneBlocking(DebugFrameDropCounter):
         def __init__(self):
             self.log = dict()
             self.lock = threading.Lock()

         def log_framedrop(self, location: string):
             if self.lock.acquire(blocking=False):
                 try:
                     self.log[location] = self.log.get(location, 0) + 1
                 finally:
                     self.lock.release()
             else:
                 pass

class DebugManager(DebugManagerBase):
    def __init__(self, debug_show_frames, debug_log_dropped_frames: FrameDropLoggingMode):
        self.frame_presenter = DebugFramePresenterFactory.get(debug_show_frames)
        self.framedrop_counter = DebugFrameDropCounterFactory.get(debug_log_dropped_frames)

    def render(self, frame, landmarks):
        self.frame_presenter.render(frame, landmarks)

    def show(self, frame):
        self.frame_presenter.draw(frame)

    def close(self):
        self.frame_presenter.close()

    def log_framedrop(self, location):
        self.framedrop_counter.log_framedrop(location)

    def print_framedrops(self):
        self.framedrop_counter.printAll()

class NoDebugManager(DebugManagerBase):

    def render(self, frame, landmarks):
        pass

    def show(self, frame):
        pass

    def close(self):
        pass

    def log_framedrop(self, param):
        pass

    def print_framedrops(self):
        pass

#static because of easier calls as the Debugmanager gets used in various components and is only needed once
debug_manager: DebugManagerBase = DebugManager(DEBUG_SHOW_IMAGE, DEBUG_SHOW_NUM_FRAMES_DROPPED) if DEBUG else NoDebugManager()