import queue
import sys
import threading
import time

import cv2
from sympy.codegen import Print

from Config import IMAGE_SOURCE, SKIP_N_FRAMES
from Utility.Exceptions import UnsuccessfulCaptureException


class FrameCapturer:
    frame_queue = queue.Queue(maxsize=3)  # own buffer to make the capture process non blocking as it is the bottleneck of the pipeline (opencv buffer not controllable in that manner)
    frame_index = 0
    """
    Used to capture a image by the usage of cv2.
    """
    def __init__(self,stop_event, image_source=IMAGE_SOURCE):
        self.frame_count = 0  # DEBUG
        self.stop = stop_event

        # set camera and frame dimensions
        self.cap = cv2.VideoCapture(image_source)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.queue_full_failures = 0 #No custom failure handle because of potential performance slowdown
        self.capture_failures = 0

    def measure_camera_fps(self,num_frames=60):
        import time
        t0 = time.time()
        for _ in range(num_frames):
            self.cap.read()
        t1 = time.time()
        return num_frames / (t1 - t0)

    def run(self):
        while not self.stop.is_set():
            try:
                frame = self.capture()
                self.frame_queue.put(frame, block=False)
            except Exception as e:
                if isinstance(e, UnsuccessfulCaptureException):
                    self.capture_failures += 1
                elif isinstance(e, queue.Full):
                    self.queue_full_failures += 1
        self.cap.release()

    def start(self):
        capture_thread = threading.Thread(target=self.run, daemon=True)
        capture_thread.start()
        return capture_thread


    def get(self):
        return self.frame_queue.get(block=True, timeout=1)


    def capture(self):
        """
        Capture and returns an image-frame.

        Raises:
            UnsuccessfulCaptureException: If the capture fail

        Return:
            np.ndarray: RGB image as a 3-dimensional (height,width, color-channels) NumPy array.
        """
        successful_read, frame = None,None
        for i in range(-1,SKIP_N_FRAMES): #only reliable way to skip frames and therefore limit to a specific refreshrate
            successful_read, frame = self.cap.read()

        if frame is None or not successful_read:
            raise UnsuccessfulCaptureException(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB colorscheme needed for landMarc extraction