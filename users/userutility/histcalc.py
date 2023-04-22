import matplotlib.pyplot as plt
import cv2
import numpy as np

im = cv2.imread('c6.jpg')
# calculate mean value from RGB channels and flatten to 1D array
bright_count = np.sum(np.array(im) >= 200)
print("Birght Pixel Count =", bright_count)
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0, 255])
plt.show()
#plt.hist(im.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram
#plt.show()

# load an image in grayscale mode
img = cv2.imread('c6.jpg', 0)

# calculate frequency of pixels in range 0-255
histg = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(histg)
plt.show()

plt.hist(img.ravel(),256,[0,256])
plt.show()