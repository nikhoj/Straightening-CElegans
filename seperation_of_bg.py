import random

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def seperation_of_bg():
    dir = 'frames'
    img_names = os.listdir(dir)

    r = random.randint(100, len(img_names) - 100)

    img = cv2.imread(dir + '/' + img_names[r], cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('frame.jpg', img)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # plt.title('actual image')
    # plt.imshow(img, cmap='gray')
    # plt.show()

    mask_image = cv2.imread('mean_image.jpg', cv2.IMREAD_GRAYSCALE)
    plt.title('mask image')
    plt.imshow(mask_image, cmap='gray')
    plt.show()

    mask = mask_image < 155
    # plt.title('mask')
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    img3 = img - (1 - mask) * 255

    # img3 = img3 - np.min(img3)

    normalized_image = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX)
    plt.title('after delete the masked pixels')
    plt.imshow(1 - normalized_image, cmap='gray')
    plt.show()
    cv2.imwrite('normalized_image.jpg', normalized_image)

    # otsu threshold
    # divide
    # blur
    img = cv2.imread('normalized_image.jpg', cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), sigmaX=0, sigmaY=0)
    plt.title('blur')
    plt.imshow(blur, cmap='gray')
    plt.show()

    divide = cv2.divide(img, blur, scale=255)

    plt.title('divide')
    plt.imshow(divide, cmap='gray')
    plt.show()

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    plt.title('thresh')
    plt.imshow(thresh, cmap='gray')
    plt.show()

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    plt.title('morph')
    plt.imshow(morph, cmap='gray')
    plt.show()

    kernel = np.ones((2, 2))
    convolved_image = cv2.filter2D(thresh, -1, kernel)

    plt.title('convolved_image')
    plt.imshow(convolved_image, cmap='gray')
    plt.show()
    cv2.imwrite('conv_img.jpg', convolved_image)
