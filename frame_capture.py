import cv2
import numpy as np
import os


# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1
    init = 0
    while count < 280:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('frames/frame{}.jpg'.format(count), image)
        image = np.expand_dims(image, 0)

        if init:
            image_tensor = np.concatenate((image_tensor, image), axis=0)
        else:
            image_tensor = image

            init = 1
        count += 1

    mean_image = np.mean(image_tensor, axis=0)

    cv2.imwrite('mean_image.jpg', mean_image)
    return mean_image


# Driver Code
# if __name__ == '__main__':
#     FrameCapture("BZ33C_Chip1D_Worm27.avi")
