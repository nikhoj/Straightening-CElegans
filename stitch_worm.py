import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

folder = 'temp_cuts'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


dir = 'worm_cuts'
images = os.listdir(dir)


def stich_images(image1, image2, idx2):
    """

    :param image1: path to first image to stitch
    :param image2: path to second image to stitch
    :return: numpy_array or image which will be stitched by image1 and image2
    """
    print(image1)
    print(image2)

    image1 = cv2.imread(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread(image2, cv2.COLOR_BGR2GRAY)
    rows, cols = image1.shape[0] + image2.shape[0], max(image1.shape[1], image2.shape[1])
    frame = np.zeros((rows, cols))

    if cols == image1.shape[1]:

        startp = (cols - image2.shape[1]) // 2
        frame[0:image1.shape[0], 0:cols] = image1
        midcol_img1 = image1.shape[1] // 2
        temp = image1[:, midcol_img1]
        for i in range(image1.shape[0] - 1, 0, -1):
            if temp[i] + temp[i - 1] > 0:
                stitch_top_midp = i - 0
                break
        midcol_img2 = image2.shape[1] // 2
        temp = image2[:, midcol_img2]
        for i in range(image2.shape[0]):
            if temp[i] + temp[i - 1] > 0:
                stitch_btm_midp = i + 0
                break
        startq = stitch_top_midp - stitch_btm_midp
        for i in range(image2.shape[0]):
            for j in range(image2.shape[1]):
                frame[startq + i, j + startp] = max(image2[i, j], frame[startq + i, j])
        j = rows - 1
        while np.sum(frame[j, :]) == 0:
            j = j - 1

        frame = frame[:j+1, :]
    else:
        # print(cols - image1.shape[1])
        startp = (cols - image1.shape[1]) // 2
        frame[0:image1.shape[0], startp:image1.shape[1] + startp] = image1
        midcol_img1 = image1.shape[1] // 2
        temp = image1[:, midcol_img1]
        for i in range(image1.shape[0] - 1, 0, -1):
            if temp[i] + temp[i - 1] > 0:
                stitch_top_midp = i - 0
                break
        midcol_img2 = image2.shape[1] // 2
        temp = image2[:, midcol_img2]
        for i in range(image2.shape[0]):
            if temp[i] + temp[i - 1] > 0:
                stitch_btm_midp = i + 0
                break
        startq = stitch_top_midp - stitch_btm_midp
        for i in range(image2.shape[0]):
            for j in range(image2.shape[1]):
                frame[startq + i, j] = max(image2[i, j], frame[startq + i, j])

        j = rows - 1
        while np.sum(frame[j, :]) == 0:
            j = j - 1

        frame = frame[:j + 1, :]

    # print(startp)
    # plt.imshow(frame)
    # plt.show()

    cv2.imwrite('temp_cuts/worm_cut_{}.jpg'.format(idx2), frame)

idx = 0
stich_images(dir + '/worm_cut_{}.jpg'.format(idx), dir + '/worm_cut_{}.jpg'.format(idx + 1), 100*(idx+1))

for idx in range(1,len(images)-1):
    dir2 = 'temp_cuts'
    stich_images(dir2 + '/worm_cut_{}.jpg'.format(idx*100), dir + '/worm_cut_{}.jpg'.format(idx + 1), 100*(idx+1))
