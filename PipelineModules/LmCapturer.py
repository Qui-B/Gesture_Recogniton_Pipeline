import cv2

from Exceptions import UnsuccessfulCaptureException
from Config import IMAGE_SOURCE, FPS


class LmCapturer:
    """
    Used to capture a image by the usage of cv2.
    """
    def __init__(self, image_source=IMAGE_SOURCE):
        self.frame_count = 0  # DEBUG

        # set camera and frame dimensions
        self.cap = cv2.VideoCapture(image_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        if FPS <= 0 or FPS >= self.cap.get(cv2.CAP_PROP_FPS):
            self.take_n_frame = 1
        else:
            self.take_n_frame = int(self.cap.get(cv2.CAP_PROP_FPS) / FPS) #For skipping frames and therefore limiting refreshrate



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

        if not successful_read or frame is None:
            raise UnsuccessfulCaptureException(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #cv2.imshow(IMAGE_SOURCE, frame)
        #cv2.waitKey(1)
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB colorscheme needed for landMarc extraction
        return RGB_frame

    """
        def capture(self):
        Capture and returns an image-frame.

        Raises:
            UnsuccessfulCaptureException: If the capture fail

        Return:
            np.ndarray: RGB image as a 3-dimensional (height,width, color-channels) NumPy array.

        successful_read, frame = None
        for i in range(0,self.num_frame_skips):
            successful_read, frame = self.cap.read()
        if not successful_read or frame is None:
            raise UnsuccessfulCaptureException(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #DEBUGGING-----------------------------------------------------------
        if self.frame_count < self.num_frame_skips: #DEBUG
            self.frame_count = self.frame_count + 1
            return None
        self.frame_count = 0
        cv2.imshow(IMAGE_SOURCE, frame)
        cv2.waitKey(1)
        #---------------------------------------------------------------------------------
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB colorscheme needed for landMarc extraction
        return RGB_frame
    """