import cv2
import numpy as np

# Load the image
image = cv2.imread('masked_worm.jpg', 0)


# Apply thresholding to the grayscale image
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the thresholded image
contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding box of the largest contour
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Extract the region of interest from the original image
roi = image[y:y+h, x:x+w]

# Define the rotation angle and pivot point
angle = 45
pivot = (w//2, h//2)

# Define the rotation matrix
M = cv2.getRotationMatrix2D(pivot, angle, 1)

# Apply the rotation to the region of interest
rotated_roi = cv2.warpAffine(roi, M, (w, h))

# Replace the region of interest with the rotated image
image[y:y+h, x:x+w] = rotated_roi

# Display the result
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
