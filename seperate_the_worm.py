import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

frame = cv2.imread('frame.jpg', cv2.COLOR_BGR2GRAY)
mask = cv2.imread('masked_worm.jpg', cv2.COLOR_BGR2GRAY)
# plt.title('frame')
# plt.imshow(frame, cmap='gray')
# plt.show()

maskbin = mask > 125
worm_img = maskbin * frame
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
rotated_image = cv2.warpAffine(worm_img[200:-200, 200:-200], rotation_matrix, (frame.shape[1], frame.shape[0]))


# plt.title('frame')
# plt.imshow(rotated_image, cmap='gray')
# plt.show()

# lets find the backbone set

def find_backbone(head_x, head_y, maskbin):
    backbone = [(head_x, head_y)]
    for h in range(head_x + 1, tail_x + 1):
        alist = []
        for w in range(W):
            if maskbin[h, w]:
                alist.append(w)
        temp_y = int(np.mean(alist))
        backbone.append((h, temp_y))

    # print(backbone)
    # print(len(backbone))
    A = np.array(backbone)
    # print(A)
    # plt.figure()
    # plt.scatter(x=A[:,0], y = A[:,1])
    # plt.show()

    return A


# def find_next_rotation_point(A):
#     # Calculate the numerical derivative
#     dydx2 = np.diff(np.diff(A[:, 1]) / np.diff(A[:, 0]))
#
#     # Print the result
#
#     plt.figure()
#     plt.plot(dydx2)
#     plt.show()
#
#     dydx2 = np.abs((dydx2))
#     # print(dydx2)
#
#     # rotation from here
#
#     listofpoints = [(head_x, head_y)]
#     start_x, start_y = head_x, head_y
#     print('rotate start from ({},{})'.format(start_x, start_y))
#     for i in range(2, len(dydx2) - 1, 2):
#         if dydx2[i] > 2:
#             # print('rotate at location to ({},{})'.format(start_x + i, A[i, 1]))
#             listofpoints.append((start_x + i, A[i, 1]))
#     return listofpoints


# def rotate2D(head_x, head_y, tail_x, tail_y, image, p):
#     # Angle between the head and tail fixing here
#     theta = (head_y - tail_y) / (head_x - tail_x)
#     theta = theta * 180 / np.pi
#     print(theta)
#
#     width = tail_y - head_y
#     height = tail_x - head_x
#
#     print(width, height)
#
#     if width < height:  # I have to cross-check the logic
#         theta = -theta
#     else:
#         theta = 90 - theta
#
#     pivot = (int(head_x), int(head_y))
#
#     # Create a rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D(pivot, theta, scale=1.0)
#
#     # Apply the rotation to the image
#     rotated_image = cv2.warpAffine(worm_img[200:-200,200:-200], rotation_matrix,
#                                    (worm_img.shape[1], worm_img.shape[0]))
#
#     plt.title('After rotation')
#     plt.imshow(255 - rotated_image, cmap='gray')
#     plt.show()
#
#     maskbin = rotated_image > 0
#
#     H, W = worm_img.shape
#     print(H, W)
#
#     for h in range(H):
#         # find the head of the worm
#         if np.sum(maskbin[head_x:,:][h, :]) > 0:
#             new_head_x = h
#             print('found the head of the worm at x-axis {}'.format(h))
#             break
#
#     # for h in range(H - 1, 0, -1):
#     #     # find the head of the worm
#     #     if np.sum(maskbin[h, :]) > 0:
#     #         new_tail_x = h
#     #         print('found the tail of the worm at x-axis {}'.format(h))
#     #         break
#
#     # find the head tail coordinate
#
#     list_of_y = []
#     for w in range(W):
#         if maskbin[head_x:,:][new_head_x, w]:
#             list_of_y.append(w)
#
#     new_head_y = int(np.median(list_of_y))
#
#     # # find the head tail coordinate
#     #
#     # list_of_y = []
#     # for w in range(W):
#     #     if maskbin[new_tail_x, w]:
#     #         list_of_y.append(w)
#     #
#     # new_tail_y = int(np.median(list_of_y))
#
#     M = np.float32([
#         [1, 0, head_y - new_head_y],
#         [0, 1, head_x - new_head_x],
#     ])
#
#     shifted = cv2.warpAffine(rotated_image, M, rotated_image.shape[::-1])
#     plt.title('After shifted')
#     plt.imshow(255 - shifted, cmap='gray')
#     plt.show()
#
#     cv2.imwrite('worm_cuts/cropped_{}.jpg'.format(p), 255-shifted[head_x:new_head_x,:])
#
#     #maskbin = worm_img > 0
#
#     #return worm_img, maskbin
#
#
# A = find_backbone(head_x, head_y, maskbin)
# control_points = find_next_rotation_point(A)
# for i in range(len(control_points)-1):
#     x1, y1 = control_points[i]
#     x2, y2 = control_points[i+1]
#     #worm_img, maskbin = rotate2D(x1, y1, x2, y2, worm_img, 0)
#     #worm_img, maskbin = rotate2D(x1, y1, x2, y2, worm_img, 1)
#     rotate2D(x1, y1, x2, y2, worm_img, i)
