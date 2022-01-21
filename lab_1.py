import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.util import random_noise
from skimage.transform import rescale

def task1():
    bright = cv2.imread('Data/6.jpg')
    dark = cv2.imread('Data/7.jpg')

    brightIAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

    brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    darKYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

    brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
    darklSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)

    bgr = [40, 158, 16]
    thresh = 40

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(bright, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)


    hsv = cv2.cvtColor(np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

    miniSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    maxiSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    maskHSV = cv2.inRange(brightHSV, miniSV, maxiSV)
    resuleHSV = cv2.bitwise_and(brightHSV, brightHSV, mask = maskHSV)


    ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

    minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
    maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])

    maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
    resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)



    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

    minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

    maskLAS = cv2.inRange(brightIAB, minLAB, maxLAB)
    resultiAB = cv2.bitwise_and(brightIAB, brightIAB, mask=maskLAS)

    cv2.imwrite("Data/res1_bgr.png", resultBGR)
    cv2.imwrite("Data/res1_hsv.png", resuleHSV)
    cv2.imwrite("Data/res1_ycb.png", resultYCB)
    cv2.imwrite("Data/res1_lab.png", resultiAB)
def task2():


    img1 = cv2.imread('Data/1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('Data/2.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.subplot(2, 2, 3)
    plt.hist(img1.ravel(), 256, [0, 256])
    plt.xlabel("Intensity, (0..255)")
    plt.ylabel('Count, pcs')
    plt.title('Histogram of image')

    plt.subplot(2, 2, 4)

    plt.hist(img2.ravel(), 256, [0, 256])
    plt.xlabel('Intensity, (0..255)')
    plt.ylabel('Count, pcs')
    plt.title('Histogram of image')

    plt.show()

def task3():
    img1 = cv2.imread('Data/1.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = rescale(img1, 0.15)

    img_ns = random_noise(img1, mode='speckle', mean=0.7)

    img_m3 = median(img_ns, disk(3))
    img_m9 = median(img_ns, disk(11))

    plt.subplot(2,2,1)
    plt.imshow(img1, cmap = 'gray')
    plt.title('Original image')
    plt.subplot(2,2,2)
    plt.imshow(img_ns, cmap=  'gray')
    plt.title("Noisy image")
    plt.subplot(2, 2, 3)
    plt.imshow(img_m3, cmap='gray')
    plt.title("Median filter (r =3)")
    plt.subplot(2, 2, 4)
    plt.imshow(img_m9, cmap='gray')
    plt.title("Median filter (r = 9)")
    plt.show()

    img_g1 = gaussian(img_ns, 1)
    img_g3 = gaussian(img_ns, 3)

    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original image')
    plt.subplot(2, 2, 2)
    plt.imshow(img_ns, cmap='gray')
    plt.title("Noisy image")
    plt.subplot(2, 2, 3)
    plt.imshow(img_g1, cmap='gray')
    plt.title("Gaussian filter (sigma = 1)")
    plt.subplot(2, 2, 4)
    plt.imshow(img_g3, cmap='gray')
    plt.title("Gaussian filter (sigme = 3)")
    plt.show()

if __name__ == "__main__":
    task1()
    task3()