# import the necessary packages
import matplotlib.pyplot as plt
import cv2
import numpy as np


def threshold_slow(T, image):
    # grab the image dimensions
    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            image[y, x] = 255 if image[y, x] >= T else 0

    # return the thresholded image
    return image


# load the original image, convert it to grayscale, and display
# it inline
image = cv2.imread("/2020 Datasets/corona R and D/cdischarge/c12.jpg")
img = image
print("Shape ", image.shape)
avg_color_per_row = np.average(image, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print("Color info=", avg_color)
from matplotlib import pyplot as plt

# img = cv2.imread('p.jpg',0)

# alternative way to find histogram of an image
plt.hist(image.ravel(), 256, [0, 256])
plt.show()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
a = np.array(image)
print("BrightNess=", a.max(), np.unravel_index(a.argmax(), a.shape))
plt.imshow(image, cmap="gray")
plt.show()
arr = np.asarray(image)
print(len(arr))
#####################################################################
# Get Oriantations
hh, ww, cc = img.shape

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image
ret, thresh = cv2.threshold(gray, 0, 255, 0)

# find outer contour
cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

# get rotated rectangle from outer contour
rotrect = cv2.minAreaRect(cntrs[0])
box = cv2.boxPoints(rotrect)
box = np.int0(box)

# draw rotated rectangle on copy of img as result
result = img.copy()
cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

# get angle from rotated rectangle
angle = rotrect[-1]

# from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
print("Orinatations is ", angle)
if angle < -45:
    angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = -angle

print(angle, "deg")

# write result to disk
#cv2.imwrite("Resultimage.png", result)

cv2.imshow("THRESH", thresh)
cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
##########################################################
# arr = np.split(arr, 20)
# arr = np.array([np.split(x, 20, 1) for x in arr])
# print("Arr Values ",arr)
image = threshold_slow(5, image)
plt.imshow(image, cmap="gray")
plt.show()

#############################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/2020 Datasets/corona R and D/cdischarge/c12.jpg')  # you can read in images with opencv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_color1 = np.asarray([0, 0, 255])  # white!
hsv_color2 = np.asarray([30, 255, 255])  # yellow! note the order

mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

plt.imshow(mask, cmap='gray')  # this colormap will display in black / white
plt.show()

