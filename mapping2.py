import math
import imgstitch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import shutil

folder = 'worm_cuts'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))



frame = cv2.imread('frame.jpg', cv2.COLOR_BGR2GRAY)
mask = cv2.imread('masked_worm.jpg', cv2.COLOR_BGR2GRAY)
maskbin = mask == 255
worm_img = maskbin * frame


# Now lets cut the worm into square pieces and keep the one which has worm only

def get_headtail(img):
    H, W = img.shape
    print(H, W)

    for h in range(H):
        # find the head of the worm
        if np.sum(img[h, :]) > 0:
            head_x = h
            print('found the head of the worm at x-axis {}'.format(h))
            break

    for h in range(H - 1, 0, -1):
        # find the head of the worm
        if np.sum(img[h, :]) > 0:
            tail_x = h
            print('found the tail of the worm at x-axis {}'.format(h))
            break

    # find the head tail coordinate

    list_of_y = []
    for w in range(W):
        if img[head_x, w]:
            list_of_y.append(w)

    head_y = int(np.median(list_of_y))

    # find the head tail coordinate

    list_of_y = []
    for w in range(W):
        if img[tail_x, w]:
            list_of_y.append(w)

    tail_y = int(np.median(list_of_y))

    print('Head is at ({},{}), and tail is at ({},{}).'.format(head_x, head_y, tail_x, tail_y))

    return head_x, head_y, tail_x, tail_y


head_x, head_y, tail_x, tail_y = get_headtail(maskbin)


# #Angle between the head and tail fixing here
# theta = math.degrees(math.atan((head_y - tail_y)/(head_x - tail_x)))
#
# width = tail_y - head_y
# height = tail_x - head_x
#
# print(width, height)
#
# if width < height:  # I have to cross-check the logic
#     theta = -theta
# else:
#     theta = 90 - theta
#
# # Create a rotation matrix
# rotation_matrix = cv2.getRotationMatrix2D((head_x, head_y), theta, scale=1.0)
#
# # Apply the rotation to the image
# worm_img = cv2.warpAffine(worm_img, rotation_matrix, (worm_img.shape[1], worm_img.shape[0]))
#
# plt.title('worm image rotation fix')
# plt.imshow(worm_img, cmap='gray')
# plt.show()
#
# plt.title('maskbin')
# plt.imshow(maskbin, cmap='gray')
# plt.show()
# maskbin = cv2.warpAffine(maskbin, rotation_matrix, (maskbin.shape[1], maskbin.shape[0]))
# plt.title('mask image rotation fix')
# plt.imshow(maskbin, cmap='gray')
# plt.show()
# find the backbone set

def find_backbone(head_x, head_y, img):
    H, W = img.shape
    backbone = [(head_x, head_y)]
    for h in range(head_x + 1, tail_x + 1):
        alist = []
        for w in range(W):
            if img[h, w]:
                alist.append(w)
        temp_y = int(np.mean(alist))
        backbone.append((h, temp_y))

    A = np.array(backbone).reshape((len(backbone), 2))

    return A


A = find_backbone(head_x, head_y, maskbin)

Amap = np.zeros((len(A), 1))

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

cutpts_x = np.append(A[::10, 0], A[-1, 0])
cutpts_y = np.append(A[::10, 1], A[-1, 1])
cutpts = list(np.dstack([cutpts_x.flat, cutpts_y.flat])[0])

# dydx = np.gradient(A[:, 1], A[:, 0])  # np.diff(A[:, 1]) / np.diff(A[:, 0])
#
# plt.figure()
# # plt.plot(A[:, 0][0::30], A[:, 1][::30], 'o--', label='worm')
# plt.plot(A[:, 0], dydx, 'o--', label='derivative')
# plt.legend()
# plt.show()
#
# cutpts = [(A[0, 0], A[0, 1])]
# for i in range(len(dydx)):
#     if np.abs(dydx[i]) > 6:
#         print(dydx[i], i + A[0, 0], A[i, 1])
#         if (i + A[0, 0]) - cutpts[-1][0] >= 10:
#             cutpts.append((i + A[0, 0], A[i, 1]))
# cutpts.append((A[-1, 0], A[-1, 1]))
# print(cutpts)


# # --------------------------------------------------#
# cutpts_x = np.append(A[0::40, 0], A[-1, 0])
# cutpts_x = np.append(cutpts_x, A[0, 0] - 160)
# cutpts_x = np.append(cutpts_x, max(cutpts_x) + 160)
# cutpts_y = np.append(A[0::40, 1], A[-1, 1])
# cutpts_y = np.append(cutpts_y, A[0, 1])
# cutpts_y = np.append(cutpts_y, A[-1, 1])
# cutpts = np.dstack([cutpts_x.flat, cutpts_y.flat])[0]
# # --------------------------------------------------#
# for pts in cutpts:
#     cutpts = np.vstack((cutpts, np.array([[pts[0], pts[1] + 160]])))
#     cutpts = np.vstack((cutpts, np.array([[pts[0], pts[1] - 160]])))
#     cutpts = np.vstack((cutpts, np.array([[pts[0], pts[1] + 2 * 160]])))
#     cutpts = np.vstack((cutpts, np.array([[pts[0], pts[1] - 2 * 160]])))
#
# fig, ax = plt.subplots()
# ax.imshow(worm_img, cmap='gray')
# ax.plot(cutpts[:, 1], cutpts[:, 0], '.b')
# ax.axis((0, worm_img.shape[1], worm_img.shape[0], 0))
# plt.show()
#
# # --------------------------------------------------#
# dstpts_x = np.append(A[0::40, 2], A[-1, 2])
# dstpts_x = np.append(dstpts_x, A[0, 2] - 160)
# dstpts_x = np.append(dstpts_x, max(dstpts_x) + 160)
# dstpts_y = np.array([A[0, 1]] * len(dstpts_x))
# dstpts = np.dstack([dstpts_x.flat, dstpts_y.flat])[0]
# # --------------------------------------------------#
#
# for pts in dstpts:
#     dstpts = np.vstack((dstpts, np.array([[pts[0], pts[1] + 160]])))
#     dstpts = np.vstack((dstpts, np.array([[pts[0], pts[1] - 160]])))
#     dstpts = np.vstack((dstpts, np.array([[pts[0], pts[1] + 2 * 160]])))
#     dstpts = np.vstack((dstpts, np.array([[pts[0], pts[1] - 2 * 160]])))
#
# fig, ax = plt.subplots()
# ax.imshow(worm_img, cmap='gray')
# ax.plot(dstpts[:, 1], dstpts[:, 0], '.b')
# ax.axis((0, worm_img.shape[1], worm_img.shape[0], 0))
# plt.show()

frame2 = np.zeros(maskbin.shape)
wn = 0
while len(cutpts) > 1:
    temp_frame = frame2.copy()
    pts = cutpts.pop(0)
    x1, y1 = pts
    x2, y2 = cutpts[0]
    temp_frame[int(x1):int(x2)] = worm_img[int(x1):int(x2)]
    # plt.title('cut')
    # plt.imshow(temp_frame)
    # plt.show()

    # center point
    cx, cy = temp_frame.shape

    angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))

    if y2 > y1 or angle < 90:  # I have to cross-check the logic
        angle = -angle
    else:
        angle = 90 - angle

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cx // 2, cy // 2), angle, scale=1.0)
    R = rotation_matrix[:, 0:2]

    # src_pt1 = np.array([0, 0, 1])
    # src_pt2 = np.array([382, 1399, 1])
    #
    # dst_pt1 = np.matmul(rotation_matrix, src_pt1)
    # dst_pt2 = np.matmul(rotation_matrix, src_pt2)

    rotated_image = cv2.warpAffine(temp_frame, rotation_matrix, (temp_frame.shape[1], temp_frame.shape[0]))

    # plt.title('rotated at {} angle'.format(angle))
    # plt.imshow(rotated_image)
    # plt.show()

    # now get only the meat slice
    v, x, y, z = get_headtail(rotated_image)
    H, W = rotated_image.shape
    print(H, W)

    for w in range(W):
        # find the head of the worm
        if np.sum(rotated_image[:, w]) > 0:
            col_start = w

            break
    for w in range(W-1,0,-1):
        # find the head of the worm
        if np.sum(rotated_image[:, w]) > 0:
            col_end = w

            break

    cv2.imwrite('worm_cuts/worm_cut_{}.jpg'.format(wn), rotated_image[min(v,y):max(v,y),col_start:col_end])
    wn += 1

# src = A[:,0:2]
#
# dst = src.copy()
# dst[:,1] = src[0,1]
# dst[:,0] = A[:,2]

# M = cv2.getAffineTransform(src[0::25], dst[0::25])
# dst = cv2.warpAffine(worm_img, M, (worm_img.shape[1], worm_img.shape[0]))

# plt.subplot(121)
# plt.imshow(img)
# plt.title('Input')
#
# plt.subplot(122)
# plt.imshow(dst)
# plt.title('Output')
#
# plt.show()

# from skimage.transform import PiecewiseAffineTransform, warp
#
#
#
# rows = int(max(dstpts[:, 0])) - int(min(dstpts[:, 0]))
# cols = int(max(cutpts[:, 1])) - int(min(cutpts[:, 1]))
#
# plt.imshow(worm_img[int(min(dstpts[:, 0])):int(min(dstpts[:, 0])) + rows,
#            int(min(dstpts[:, 1])): int(min(dstpts[:, 1])) + cols], cmap='gray')
# plt.show()
#
# a = int(min(dstpts[:, 0]))
# b = int(min(dstpts[:, 0])) + rows
# c = int(min(dstpts[:, 1]))
# d = int(min(dstpts[:, 1])) + cols
#
# cutpts[:, 0] = cutpts[:, 0] - min(cutpts[:, 0])
# cutpts[:, 1] = cutpts[:, 1] - min(cutpts[:, 1])
#
# fig, ax = plt.subplots()
# ax.imshow(worm_img[a:b, c:d], cmap='gray')
# ax.plot(cutpts[:, 1], cutpts[:, 0], '.r')
# ax.axis((0, cols, rows, 0))
# plt.show()
#
# dstpts[:, 0] = dstpts[:, 0] - min(dstpts[:, 0])
# dstpts[:, 1] = dstpts[:, 1] - min(dstpts[:, 1])
#
# fig, ax = plt.subplots()
# ax.imshow(worm_img[a:b, c:d], cmap='gray')
# ax.plot(dstpts[:, 1], dstpts[:, 0], '.r')
# ax.axis((0, cols, rows, 0))
# plt.show()
#
# # plt.imshow(worm_img[int(min(dstpts[:, 0])):int(min(dstpts[:, 0])) + rows, int(min(dstpts[:, 1])): int(min(dstpts[:, 1])) + cols ], cmap='gray')
# # plt.show()
#
# tform = PiecewiseAffineTransform()
# tform.estimate(cutpts, dstpts)
# out = warp(worm_img[a:b, c:d], tform, output_shape=(rows, cols))
# out = warp(out, tform, output_shape=(rows, cols))
# out = warp(out, tform, output_shape=(rows, cols))
#
# fig, ax = plt.subplots()
# ax.imshow(255-out, cmap='gray')
# # ax.plot(tform.inverse(cutpts)[:, 0], tform.inverse(cutpts)[:, 1], '.b')
# # ax.axis((0, cols, rows, 0))
# plt.show()

# dir = 'worm_cuts'
# files = os.listdir(dir)
# ret = imgstitch.stitch_images('worm_cuts', files, 0)
