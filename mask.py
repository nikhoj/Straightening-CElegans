import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray


dir = 'source_images'
img_names = os.listdir(dir)

img = imread(dir + '/' + img_names[25])
gray_painting = rgb2gray(img)[:,142:710]

binarized = gray_painting < 0.55
plt.imshow(binarized, cmap='gray')
# plt.waitforbuttonpress(0)
# plt.close()

kernel = np.ones((10,10))
print(kernel)

H, W = binarized.shape

# for h in range(H):
#     for w in range(W):
#         if binarized[h:h+9, w:w+9]:


