import  numpy as np
import cv2 as cv
from skimage import img_as_ubyte, morphology
    
def get_boundaries(image, label):
    image *= 255
    image = np.squeeze(image)
    image = image.astype(np.uint8)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

    # Extract the 2D label, multiply it by 255 so that values are
    # now in the range of 0-255, and threshold.
    label *= 255
    label = label.astype(np.uint8)
    _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)

    # Turn all 255s into 1s for the skeletonization.
    image[image == 255] = 1

    # Skeletonize the thresholded prediction and turn it back into
    # a range of 0-255.
    skel = morphology.skeletonize(image)
    skel = skel.astype(int) * 255


    image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
    label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

    # If a black image is produced then error would be inf. Place
    # a single white pixel to get a finite accuracy score.
    if (len(image_boundaries) == 0):
        image_boundaries = [0, 0]
    
    return image_boundaries, label_boundaries