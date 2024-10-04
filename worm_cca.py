import matplotlib.pyplot as plt
import numpy as np
import cv2

def worm_segmentation():
    image = cv2.imread('conv_img.jpg', cv2.COLOR_BGR2GRAY)
    image[image <= 125] = 0
    image[image > 125] = 255
    image = 255 - image
    L, W = image.shape

    plt.title('image in binary')
    plt.imshow(image)
    plt.show()

    # Finding number start from here
    x1 = input('Please give x value of any pixel the worm is at: ')
    y1 = input('Please give y value of that pixel the worm is at: ')
    y1 = int(y1)
    x1 = int(x1)
    #y1, x1 = 1262, 1258
    print(y1, x1)

    image_B = np.zeros(image.shape, dtype=np.uint8)  # create a same size image like the canvas
    image_B[y1, x1] = 255  # now create a dot in the canvas to start my dilation
    # Define the kernel to connect points (8-connectivity)
    kernel = np.array([[255, 255, 255],
                       [255, 255, 255],
                       [255, 255, 255]], dtype=np.uint8)

    while True:
        image_B_new = cv2.dilate(image_B, kernel, iterations=1)
        image_B_new = np.logical_and(image, image_B_new).astype(np.uint8) * 255
        if np.array_equal(image_B, image_B_new):
            break
        else:
            image_B = image_B_new

    plt.title('split')
    plt.imshow(image_B)
    plt.show()

    kernel = np.array([[0, 0, 0],
                       [0, 255, 0],
                       [255, 255, 255]], dtype=np.uint8)

    image_B = cv2.dilate(image_B, kernel, iterations=6)
    plt.title('split2')
    plt.imshow(image_B)
    plt.show()

    kernel = np.array([[255, 255, 255],
                       [0, 255, 0],
                       [0, 0, 0]], dtype=np.uint8)

    image_B = cv2.dilate(image_B, kernel, iterations=6)
    plt.title('split3')
    plt.imshow(image_B)
    plt.show()

    kernel = np.array([[255, 0, 0],
                       [255, 255, 0],
                       [255, 0, 0]], dtype=np.uint8)

    image_B = cv2.dilate(image_B, kernel, iterations=6)
    plt.title('split4')
    plt.imshow(image_B)
    plt.show()

    kernel = np.array([[0, 0, 255],
                       [0, 255, 255],
                       [0, 0, 255]], dtype=np.uint8)

    image_B = cv2.dilate(image_B, kernel, iterations=6)
    plt.title('split5')
    plt.imshow(image_B)
    plt.show()

    kernel = np.array([[0, 255, 0],
                       [255, 255, 255],
                       [0, 255, 0]], dtype=np.uint8)

    image_B = cv2.erode(image_B, kernel, iterations=20)
    plt.title('split6')
    plt.imshow(image_B)
    plt.show()
    cv2.imwrite('masked_worm.jpg', image_B)