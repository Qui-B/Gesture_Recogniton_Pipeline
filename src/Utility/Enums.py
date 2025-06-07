from enum import Enum

class Gesture(Enum):
    Nothing = 0
    ScrollUp = 1
    ScrollDown = 2
    SwipeLeft = 3
    SwipeRight = 4
    ZoomIn = 5
    ZoomOut = 6

class FrameDropLoggingMode(Enum):
    Quality = 0
    Fast = 1
    OFF = 2