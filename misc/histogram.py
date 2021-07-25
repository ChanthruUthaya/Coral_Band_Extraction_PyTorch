import os
import cv2 as cv
from glob import glob
import math
import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

dir = "D:/2D-remake/"
# dir = "D:/2D-remake/data/train/"
#dir = os.readlink('scratch')


image_path = dir + "/sub/images"
#label_path = dir+ "/crop_label_resize/"
label_path = dir+ "class_imb"

save_test_dir = dir + "/crop_label_resize/thresh/"
save_dir = dir + "/sub/"

# save_dir = os.readlink('scratch') + "/train"
# save_test_dir = os.readlink('scratch') + "/test"

def adjust_data(label):
        label = np.clip((label)*2/255.0, 0, 255)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label

labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

for i, label in enumerate(labels):

        img_file = glob(label_path + '/'+ label + '.*')[0].replace('\\', '/')

        # image_pil = Image.open(img_file)
        # print(image_pil.getextrema())
       # image_pil.show()

        image = cv.imread(img_file, 0)
        #cv.imshow("image",image)

        image = adjust_data(image)
        plt.hist(image.ravel(),2,[0,2]); plt.show()




        # cv.imwrite(save_test_dir + f"/{i}-label.png", image)
