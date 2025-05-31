import queue
import sys
import time

import cv2
from sympy.codegen import Print

from Config import IMAGE_SOURCE, FPS
from Utility.Exceptions import UnsuccessfulCaptureException


class FrameCapturer:
    """
    Used to capture a image by the usage of cv2.
    """
    def __init__(self,stop_event, image_source=IMAGE_SOURCE):
        self.frame_count = 0  # DEBUG
        self.stop = stop_event

        # set camera and frame dimensions
        self.cap = cv2.VideoCapture(image_source)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        self.droped_frames = 0

        self.frame_queue = queue.Queue(maxsize=3) #own buffer to make the capture process non blocking as it is the bottleneck of the pipeline (opencv buffer not controllable in that manner)
        if FPS <= 0 or FPS >= self.cap.get(cv2.CAP_PROP_FPS):
            self.take_n_frame = 1
        else:
            self.take_n_frame = int(self.cap.get(cv2.CAP_PROP_FPS) / FPS) #For skipping frames and therefore limiting refreshrate

    def measure_camera_fps(self,num_frames=60):
        import time
        t0 = time.time()
        for _ in range(num_frames):
            self.cap.read()
        t1 = time.time()
        return num_frames / (t1 - t0)

    def run(self):
        while not self.stop.is_set():
            frame = self.capture()
            #if self.frame_queue.qsize() == 3:
                #self.droped_frames += 1
            self.frame_queue.put(frame)
        self.cap.release()

    def get(self):
        #print("queue size: " + str(self.frame_queue.qsize()) + " frames Droped: " + str(self.droped_frames))
        return self.frame_queue.get(block=True)


    def capture(self):
        """
        Capture and returns an image-frame.

        Raises:
            UnsuccessfulCaptureException: If the capture fail

        Return:
            np.ndarray: RGB image as a 3-dimensional (height,width, color-channels) NumPy array.
        """
        successful_read, frame = None,None
        for i in range(0,self.take_n_frame): #only reliable way to skip frames and therefore limit to a specific refreshrate
            successful_read, frame = self.cap.read()

        if frame is None or not successful_read:
            raise UnsuccessfulCaptureException(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame