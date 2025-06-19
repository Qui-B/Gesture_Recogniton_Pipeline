import string
import threading
from abc import ABC, abstractmethod

from src.Utility.Enums import FrameDropLoggingMode


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