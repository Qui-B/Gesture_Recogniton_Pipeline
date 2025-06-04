import queue
import threading
import time

import cv2

from src.Config import IMAGE_SOURCE, SKIP_N_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH
from src.Utility.Exceptions import UnsuccessfulCaptureException


class FrameCapturer:
    frame_queue = queue.Queue(maxsize=3)  # own buffer to make the capture process non blocking as it is the bottleneck of the pipeline (opencv buffer not controllable in that manner)
    frame_index = 0
    """
    Used to capture a image by the usage of cv2.
    """
    def __init__(self,start_event,stop_event):
        self.stop_event = stop_event
        self.start_event = start_event

        self.cap = None
        self.capInit()

        self.queue_full_failures = 0 #No custom failure handle because of potential performance slowdown
        self.capture_failures = 0

    def measure_camera_fps(self,num_frames=60):
        self.stop_event.set()
        t0 = time.time()
        for _ in range(num_frames):
            self.capture()
        t1 = time.time()
        self.stop_event.clear()
        return num_frames / (t1 - t0)

    def capInit(self):
        # set camera and frame dimensions
        self.cap = cv2.VideoCapture(IMAGE_SOURCE)
        self.cap.set(cv2.CAP_PROP_FPS, 1000)  # helps in some cases if the camera defaults to lower fps
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


    def start(self):
        if self.cap is None:
            self.capInit()
        captureThread = threading.Thread(target=self.run)
        captureThread.start()


    def run(self):
        self.start_event.wait()
        while not self.stop_event.is_set():
            try:
                frame = self.capture()
                self.frame_queue.put(frame, block=False)
            except Exception as e:
                if isinstance(e, UnsuccessfulCaptureException):
                    self.capture_failures += 1
                elif isinstance(e, queue.Full):
                    self.queue_full_failures += 1
        self.cap.release()

    def get(self):
        return self.frame_queue.get(block=True, timeout=2) #TODO maybe add timeout


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