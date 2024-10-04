import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from frame_capture import FrameCapture
from seperation_of_bg import seperation_of_bg
from worm_cca import worm_segmentation
from mapping import straightening_worm

if __name__ == '__main__':
    FrameCapture("BZ33C_Chip1D_Worm27.avi")
    seperation_of_bg()
    worm_segmentation()
    straightening_worm()
