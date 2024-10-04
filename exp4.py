import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.transform import PiecewiseAffineTransform, warp

# get properly sorted images
dir = 'frames'
img_names = os.listdir(dir)
print(img_names)

idx = 160

img1 = cv2.imread('frames/frame' + str(idx) + '.jpg', 0)
img2 = cv2.imread('frames/frame' + str(idx + 1) + '.jpg', 0)


# Get motion mask
def get_mask(frame1, frame2, kernel=np.array((9, 9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs:
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            kernel - (NxN) array for Morphological Operations
        Outputs:
            mask - Thresholded mask for moving pixels
        """
    frame_diff = cv2.subtract(frame2, frame1)

    # blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)

    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY_INV, 11, 2)

    mask = cv2.medianBlur(mask, 3)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=20)

    return mask


kernel = np.array((7, 7), dtype=np.uint8)
mask = get_mask(img1, img2, kernel)

plt.imshow(mask, cmap='gray')
plt.title("Motion Mask")
plt.show()

image = mask
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original')

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton')

fig.tight_layout()
plt.show()

# choose the control points

H, W = mask.shape
print(H, W)

for h in range(H):
    # find the head of the worm
    if np.sum(skeleton[h, :]) > 0:
        head_x = h
        print('found the head of the worm at x-axis {}'.format(h))
        break

for h in range(H - 1, 0, -1):
    # find the head of the worm
    if np.sum(skeleton[h, :]) > 0:
        tail_x = h
        print('found the tail of the worm at x-axis {}'.format(h))
        break

# find the head tail coordinate

list_of_y = []
for w in range(W):
    if skeleton[head_x, w]:
        list_of_y.append(w)

head_y = int(np.median(list_of_y))

# find the head tail coordinate

list_of_y = []
for w in range(W):
    if skeleton[tail_x, w]:
        list_of_y.append(w)

tail_y = int(np.median(list_of_y))

print('Head is at ({},{}), and tail is at ({},{}).'.format(head_x, head_y, tail_x, tail_y))


width = np.abs(head_y-tail_y)
height = np.abs((head_x -tail_x))

print('height is {} and width is {}'.format(height, width))

if height > width:
    angle = (head_y - tail_y)/(head_x-tail_x) * (180/np.pi)
    rotation_matrix = cv2.getRotationMatrix2D((head_x,head_y), -angle, 1)
    image = cv2.warpAffine(image,rotation_matrix,image.shape[1::-1])
    #image = image[head_x-300:tail_x+300,head_y-600:tail_y + 600]
    image = np.rot90(image)

plt.imshow(image)
plt.show()

rows, cols = image.shape
src_rows = np.linspace(20,rows, 5)
src_cols = np.linspace(20,cols, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

dst_rows = np.linspace(20+rows//4, 3 * (rows//4), 5)
dst_cols1 = np.linspace(20, cols//8, 5)
dst_cols2 = np.linspace(20 + 7 * (cols//8),cols, 5)
dst_cols = np.concatenate((dst_cols1,dst_cols2), axis=0)
dst_rows, dst_cols = np.meshgrid(dst_rows, dst_cols)
dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(src[:, 0], src[:, 1], '.b')
plt.show()

fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(dst[:, 0], dst[:, 1], '.b')
plt.show()

tform = PiecewiseAffineTransform()
tform.estimate(dst, src)

out_rows = rows
out_cols = cols + 200
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()

