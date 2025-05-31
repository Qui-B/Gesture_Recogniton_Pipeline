import threading

from PipelineModules.FrameCapturer import FrameCapturer

stop_event = threading.Event
frameCapturer = FrameCapturer(stop_event)
print(frameCapturer.measure_camera_fps(1000))

