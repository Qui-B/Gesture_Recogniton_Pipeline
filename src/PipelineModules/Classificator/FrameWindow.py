
from collections import deque
import torch

#from Hand_Recognition.DebugUtil import printAverageDirectionChange, printLandMarkCoordinate
from Config import WINDOW_LENGTH, FEATURE_VECTOR_LENGTH, GCN_NUM_OUTPUT_CHANNELS, DEVICE
from Utility.Dataclasses import SpatialFeaturePackage


#Most methods corresponding to FrameWindow are in GraphTcn because of the needed preprocessing before Windowupdates
class FrameWindow:

    """
    Stores the most recent landmark vectors in a deque which serves as a input for the classification tcp. Used by GraphTCN.
    """
    def __init__(self) -> None:
        self.frame_deque = deque(maxlen=WINDOW_LENGTH)
        self.FLATTENED_T_LENGTH = FEATURE_VECTOR_LENGTH * GCN_NUM_OUTPUT_CHANNELS + 1 #need for tensor dimension remapping in update frame_deque
                                                                                      #+1 to store the hand detected flag

    def flatten2DTensor(self, spatial_feature_pkg: SpatialFeaturePackage):
        window_t = torch.zeros(self.FLATTENED_T_LENGTH, device=DEVICE)
        window_t[0] = float(spatial_feature_pkg.hand_detected)
        window_t[1:] = spatial_feature_pkg.spatial_lm_t.view(-1)  # flatten by concatenating rows (HAND_DETECTED_ELEM,x0,y0,z0,x1,y1,z1,...,x20,y20,z20)
        return window_t

    def update(self, spatial_feature_pkg: SpatialFeaturePackage):
        flattened_t = self.flatten2DTensor(spatial_feature_pkg)
        self.frame_deque.append(flattened_t)

    def getLength(self):
        return len(self.frame_deque)

    def get(self):
        return self.frame_deque

    def getAsTensor(self):
        sequence_t = (torch.stack(list(self.frame_deque))).to(DEVICE) #collect the whole frame_deque in a tensor
        sequence_t = sequence_t.unsqueeze(0) # to get a 3 dimensional vector with BATCH_SIZE  = 1
        sequence_t = sequence_t.permute(0, 2, 1)  #(1, seq_len, feature_dim) -> (1, feature_dim, seq_len)
        return sequence_t


    def set(self, new_frame_deque):
        self.frame_deque = new_frame_deque

    def clear(self):
        self.frame_deque.clear()



