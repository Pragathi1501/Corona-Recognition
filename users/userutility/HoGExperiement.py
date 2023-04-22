#importing required libraries
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2
#%matplotlib inline


#reading the image
img = imread('/2020 Datasets/corona R and D/cdischarge/c12.jpg')
imshow(img)
avg_color_per_row = np.average(img, axis=0)
avg_color = np.average(avg_color_per_row, axis=0)
print("Image Shape ",img.shape)
print("Color info=",avg_color)

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = np.array(image)
print("BrightNess=", a.max(), np.unravel_index(a.argmax(), a.shape))
print(img.shape)
#resizing image
resized_img = resize(img, (150,950))
imshow(resized_img)
print(resized_img.shape)
#creating hog features
fd, hog_image = hog(resized_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True, multichannel=True)
print("FD Shape=",fd.shape)
#print("HOG==",hog)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

ax1.imshow(resized_img, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()
