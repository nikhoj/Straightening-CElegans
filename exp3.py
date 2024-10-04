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

idx = 150

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
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
# plt.show()

# choose the control points
# Now lets cut the worm into square pieces and keep the one which has worm only
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


# for h in range(head_x, tail_x, 10):

def find_backbone(head_x, head_y, maskbin):
    backbone = [(head_x, head_y)]
    for h in range(head_x + 1, tail_x + 1):
        alist = []
        for w in range(W):
            if maskbin[h, w]:
                alist.append(w)
        temp_y = int(np.mean(alist))
        backbone.append((h, temp_y))

    A = np.array(backbone).reshape((len(backbone), 2))

    return A


A = find_backbone(head_x, head_y, skeleton)

Amap = np.zeros((len(A), 1))

# Find the length of the worm
length = 0
for i in range(0, len(A) - 1, 1):
    x1 = A[i, 0]
    x2 = A[i + 1, 0]
    y1 = A[i, 1]
    y2 = A[i + 1, 1]
    Amap[i] = int(length + head_x)

    length = length + np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

Amap[i] = int(length + head_x)

length = length + np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
Amap[i + 1] = int(length + head_x)
A = np.append(A, Amap, 1)

src2 = A[:, 0:2]
midp = A[0, 1]

x_prime = A[:, 2].reshape(len(A), 1)
y_prime = np.zeros(x_prime.shape) + midp
y_prime = y_prime.reshape(len(A), 1)
dst2 = np.append(x_prime, y_prime, 1)

rows, cols = image.shape[0], image.shape[1]

src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

# # add sinusoidal oscillation to row coordinates
# dst_cols = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
# dst_rows = src[:, 0]
# dst_rows *= 1.5
# dst_rows -= 1.5 * 50
# dst = np.vstack([dst_cols, dst_rows]).T


dst_cols = np.linspace(0, cols, 10)
dst_rows = np.linspace(0, rows, 20)
dst_rows, dst_cols = np.meshgrid(dst_rows, dst_cols)
dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

tform = PiecewiseAffineTransform()
tform.estimate(src2, dst2)
#
out_rows = image.shape[0] - 1.5 * 20
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out, cmap='gray')
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()

# plt.figure()
# plt.scatter(x=A[:,0], y = A[:,1])


dydx2 = np.diff(np.diff(A[:, 1]) / np.diff(A[:, 0]))

# Print the result

plt.figure()
plt.plot(dydx2)
plt.show()

dydx2 = np.abs((dydx2))
print(dydx2)
