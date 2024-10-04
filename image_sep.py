import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold

dir = 'source_images'
img_names = os.listdir(dir)


img = cv2.imread(dir + '/' + img_names[25], cv2.IMREAD_GRAYSCALE)
# img = img[:,142:710]
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=True)
plt.show()

# dir = 'source_images'
# img_names = os.listdir(dir)
#
# mean_image = cv2.imread(dir + '/' + img_names[0], cv2.IMREAD_GRAYSCALE)
#
# for i in range(0, len(img_names), 1):
#     image = cv2.imread(dir + '/' + img_names[i], cv2.IMREAD_GRAYSCALE)
#     # mean_image = (mean_image + image) // 2
#     #
#     # plt.imshow(image, cmap="gray")
#     # plt.waitforbuttonpress(1000)
#     # plt.close()
