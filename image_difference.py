# Script that outputs the binary differences between the stego image and the cover image. 
# All stego and cover images have the same file name and are stored in their corresponding folders.

import cv2
import numpy as np

# change path to match corresponding images you want to compare
image1 = cv2.imread('dataset/train/stego/24324.png')
image2 = cv2.imread('dataset/train/non-stego/24324.png')  

if image1 is None or image2 is None:
    print("Error: Could not load one or both images.")
    exit()

image1 = cv2.resize(image1, (128, 128))
image2 = cv2.resize(image2, (128, 128))

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

diff_image = cv2.absdiff(gray_image1, gray_image2)

_, binary_diff = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY)

# uncomment to save the image difference
# cv2.imwrite('binary_diff_image.jpg', binary_diff)

cv2.imshow('Binary Difference Image', binary_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
