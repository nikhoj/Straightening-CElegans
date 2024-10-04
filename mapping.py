import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math


def straightening_worm():
    frame = cv2.imread('frame.jpg', cv2.COLOR_BGR2GRAY)
    mask = cv2.imread('masked_worm.jpg', cv2.COLOR_BGR2GRAY)
    plt.title('frame')
    plt.imshow(frame, cmap='gray')
    plt.show()

    maskbin = mask.copy()
    worm_img = cv2.bitwise_and(frame, mask)
    plt.title('worm_img')
    plt.imshow(255 - worm_img, cmap='gray')
    plt.show()

    # Now lets cut the worm into square pieces and keep the one which has worm only
    H, W = maskbin.shape
    print(H, W)

    for h in range(H):
        # find the head of the worm
        if np.sum(maskbin[h, :]) > 0:
            head_x = h
            print('found the head of the worm at x-axis {}'.format(h))
            break

    for h in range(H - 1, 0, -1):
        # find the head of the worm
        if np.sum(maskbin[h, :]) > 0:
            tail_x = h
            print('found the tail of the worm at x-axis {}'.format(h))
            break

    # find the head tail coordinate

    list_of_y = []
    for w in range(W):
        if maskbin[head_x, w]:
            list_of_y.append(w)

    head_y = int(np.median(list_of_y))

    # find the head tail coordinate

    list_of_y = []
    for w in range(W):
        if maskbin[tail_x, w]:
            list_of_y.append(w)

    tail_y = int(np.median(list_of_y))

    print('Head is at ({},{}), and tail is at ({},{}).'.format(head_x, head_y, tail_x, tail_y))

    # Angle between the head and tail fixing here
    theta = (head_y - tail_y) / (head_x - tail_x)
    theta = theta * 180 / np.pi
    print(theta)

    width = tail_y - head_y
    height = tail_x - head_x

    print(width, height)

    if width < height:  # I have to cross-check the logic
        theta = -theta
    else:
        theta = 90 - theta

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((head_x, head_y), theta, scale=1.0)

    # Apply the rotation to the image
    worm_img = cv2.warpAffine(worm_img[200:-200, 200:-200], rotation_matrix, (frame.shape[1], frame.shape[0]))
    maskbin = cv2.warpAffine(maskbin[200:-200, 200:-200], rotation_matrix, (maskbin.shape[1], maskbin.shape[0]))

    plt.title('frame')
    plt.imshow(worm_img, cmap='gray')
    plt.show()

    plt.title('maskbin')
    plt.imshow(maskbin, cmap='gray')
    plt.show()

    # find the backbone set

    def find_backbone(head_x, head_y, tail_x, maskbin):
        backbone = [(head_x, head_y)]
        for h in range(head_x + 1, tail_x + 1):
            alist = []
            for w in range(W):
                if maskbin[h, w]:
                    alist.append(w)
            temp_y = int(np.mean(alist))
            backbone.append((h, temp_y))

        A = np.array(backbone).reshape((len(backbone), 2))
        fig, ax = plt.subplots()
        ax.set_title('backbone or skeleton')
        ax.imshow(maskbin)
        ax.plot(A[:, 1], A[:, 0])
        plt.show()

        return A

    H, W = maskbin.shape
    print(H, W)

    for h in range(H):
        # find the head of the worm
        if np.sum(maskbin[h, :]) > 0:
            head_x = h
            print('found the head of the worm at x-axis {}'.format(h))
            break

    for h in range(H - 1, 0, -1):
        # find the head of the worm
        if np.sum(maskbin[h, :]) > 0:
            tail_x = h
            print('found the tail of the worm at x-axis {}'.format(h))
            break

    # find the head tail coordinate

    list_of_y = []
    for w in range(W):
        if maskbin[head_x, w]:
            list_of_y.append(w)

    head_y = int(np.median(list_of_y))

    A = find_backbone(head_x, head_y, tail_x, maskbin)

    Amap = np.zeros((len(A), 1))
    Adeg = np.zeros((len(A), 1))

    # Find the length of the worm
    length = 0
    for i in range(len(A) - 1):
        x1 = A[i, 0]
        x2 = A[i + 1, 0]
        y1 = A[i, 1]
        y2 = A[i + 1, 1]
        Amap[i] = int(length + head_x)
        length = length + np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

    Amap[i] = int(length + head_x)
    length = length + np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
    Amap[i + 1] = int(length + head_x)

    print(length)
    A = np.append(A, Amap, 1)

    ## now creating a frame to move the flesh of the worm with respect to the skull

    frame2 = np.zeros(maskbin.shape)
    sm = 0
    mx = 0
    # Average width calculation
    for i in range(50):
        flesh = []

        pts = []
        for j in range(w):
            if maskbin[int(A[i, 0]), j]:
                flesh.append(worm_img[int(A[i, 0]), j])
        sm = sm + len(flesh)
        if len(flesh) > mx:
            mx = len(flesh)

    avg = max(sm // 50, mx)

    midy = int(A[0, 1])
    for i in range(len(A)):
        flesh = []
        pts = []
        for j in range(w):
            if maskbin[int(A[i, 0]), j]:
                flesh.append(worm_img[int(A[i, 0]), j])

        if len(flesh) > avg:
            mid_index = len(flesh) // 2
            start_point = mid_index - avg // 2
            stop_point = mid_index + avg // 2
            flesh = flesh[start_point:stop_point]
        for k in flesh:
            mid_index = len(flesh) // 2
            start_point = midy - len(flesh[:mid_index])
            frame2[int(A[i, 2]), start_point:start_point + len(flesh)] = flesh

    plt.title('frame2')
    plt.imshow(255 - frame2, cmap='gray')
    plt.show()

    # fix the row which has no pixel
    start = int(A[0, 2])
    stop = int(A[-1, 2])
    for i in range(start, stop):
        if np.sum(frame2[i, :]) == 0:
            for j in range(frame2.shape[1]):
                frame2[i, j] = (frame2[i - 2, j] + frame2[i - 1, j] + frame2[i + 1, j] + frame2[i + 2, j]) // 4

    plt.title('frame2')
    plt.imshow(255 - frame2, cmap='gray')
    plt.show()
