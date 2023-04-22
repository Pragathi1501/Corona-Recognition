from django.conf import settings
# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import hog
import cv2
from skimage.transform import resize
from skimage import exposure
import matplotlib
#matplotlib.use("TkAgg")

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
class StartProcess:
    def process(self,imagepath):
        picshape = ''
        colorinfpo = ''
        picbrightnisee = ''

        print("Image File ",imagepath)
        filepath = settings.MEDIA_ROOT + "\\datasets\\" + imagepath
        image = cv2.imread(filepath)
        resized_img = resize(image, (150, 950))
        fd, hog_image = hog(resized_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True,
                            multichannel=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        ax1.imshow(resized_img, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        plt.show()
        cv2.imshow("image", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()

        # grab the image channels, initialize the tuple of colors,
        # the figure and the flattened feature vector
        chans = cv2.split(image)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("'Flattened' Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        features = []
        # loop over the image channels
        for (chan, color) in zip(chans, colors):
            # create a histogram for the current channel and
            # concatenate the resulting histograms for each
            # channel
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)
            # plot the histogram
            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        # here we are simply showing the dimensionality of the
        # flattened color histogram 256 bins for each channel
        # x 3 channels = 768 total values -- in practice, we would
        # normally not use 256 bins for each channel, a choice
        # between 32-96 bins are normally used, but this tends
        # to be application dependent
        plt.show()
        print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
        image = cv2.imread(filepath)
        img = image
        print("Shape ", image.shape)
        picshape = image.shape
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        print("Color info=", avg_color)
        colorinfpo = avg_color
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        a = np.array(image)
        print("BrightNess=", a.max(), np.unravel_index(a.argmax(), a.shape))
        picbrightnisee = a.max()
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

        #cv2.imshow("THRESH", thresh)
        #cv2.imshow("RESULT", result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ##########################################################
        # arr = np.split(arr, 20)
        # arr = np.array([np.split(x, 20, 1) for x in arr])
        # print("Arr Values ",arr)
        image = threshold_slow(5, image)
        plt.imshow(image, cmap="gray")
        plt.show()

        return colorinfpo,picbrightnisee,picshape


