from abc import ABC, abstractmethod

from src.Config import DEBUG, DEBUG_SHOW_IMAGE, DEBUG_SHOW_NUM_FRAMES_DROPPED, DEBUG_PRINT_RESULTS
from src.Utility.DebugManager.FrameDropCounter import DebugFrameDropCounterFactory
from src.Utility.DebugManager.FramePresenter import DebugFramePresenterFactory
from src.Utility.DebugManager.ResultPrinter import DebugResultPrinterFactory
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
    def log_framedrop(self, *args):
        pass

    @abstractmethod
    def print_framedrops(self):
        pass

    @abstractmethod
    def print_result(self, *args):
        pass

class DebugManager(DebugManagerBase):
    def __init__(self,
                 debug_show_frames:  bool = DEBUG_SHOW_IMAGE,
                 debug_log_dropped_frames: FrameDropLoggingMode = DEBUG_SHOW_NUM_FRAMES_DROPPED,
                 print_results: bool = DEBUG_PRINT_RESULTS):
        self.frame_presenter = DebugFramePresenterFactory.get(debug_show_frames)
        self.framedrop_counter = DebugFrameDropCounterFactory.get(debug_log_dropped_frames)
        self.result_printer = DebugResultPrinterFactory.get(print_results)

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

    def print_result(self, confidence: float, gesture: int):
        self.result_printer.print(confidence, gesture)

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

    def print_result(self, *args):
        pass

#static because of easier calls as the DebugManager gets used in various components and is only needed once
debug_manager: DebugManagerBase = DebugManager() if DEBUG else NoDebugManager()