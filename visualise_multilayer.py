import os
import numpy as np
import cv2 as cv
from glob import glob

dir = "D:/2D-remake/3ddata/chunk1/bad_example/"

images = [os.path.splitext(file)[0] for file in os.listdir(f'{dir}/images')]

image_array = []

for i, image in enumerate(images):
    img_file = glob(f'{dir}/images/{image}.*')[0].replace('\\', '/')
    img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
    image_array.append(img)
    cv.imshow("image", img)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    print(img.shape)

image_array = np.stack(image_array, axis = 2)
cv.imwrite(f'{dir}/out.png',image_array)
print(image_array.shape)
