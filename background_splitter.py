import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dir = 'source_images'
img_names = os.listdir(dir)

image = cv2.imread(dir + '/' + img_names[10], cv2.IMREAD_GRAYSCALE)


# plt.imshow(image, cmap="gray")
# plt.waitforbuttonpress(0)
# plt.close()

def background_splitter(image):
    height, width = image.shape

    # cutting the border here
    for w in range(width):
        if np.all(image[10, w:w + 10]) > 0:
            print(w)
            image = image[:, w:]
            break

    for w in range(width):
        if np.all(image[10, w:w + 10]) == 0:
            print(w)
            image = image[:, :w]
            break

    # plt.imshow(image, cmap="gray")
    # plt.waitforbuttonpress(1000)
    # plt.close()

    # find center point
    height, width = image.shape
    center_x, center_y = int(height / 2), int(width / 2)

    print(center_x, center_y)
    image = 255 - image
    block_A = image[:center_x, :center_y]
    block_B = image[:center_x, center_y:]
    block_C = image[center_x:, :center_y]
    block_D = image[center_x:, center_y:]

    plt.imshow(block_A, cmap="gray")
    plt.waitforbuttonpress(0)
    plt.close()
    plt.imshow(block_B, cmap="gray")
    plt.waitforbuttonpress(1000)
    plt.close()
    plt.imshow(block_C, cmap="gray")
    plt.waitforbuttonpress(1000)
    plt.close()
    plt.imshow(block_D, cmap="gray")
    plt.waitforbuttonpress(1000)
    plt.close()

    return image, block_A, block_B, block_C, block_D


_, _, _, _, _ = background_splitter(image)
