import os
import argparse
import cv2 as cv
import numpy as np
from glob import glob
import math
from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default="./data", help="Data directory")
parser.add_argument("--labelprefix", type=str, default="_labels", help="prefix for saved label data")
parser.add_argument("--brightness", type=float, default=1.1, help="brightness multiplier")
parser.add_argument("--scale", type=float, default=0.5, help="amount to scale image by")

args = parser.parse_args()

"""
variables:
-- data_dir
-- label_prefix
-- scale_percent
-- brightness_factor
-- crop x1,x2,y1,y2
-- save
"""

train_root = args.dir + '/train_images/'
test_root = args.dir + '/test_images/'

train_path = train_root + "/images"
train_label_path = train_root + "/labels"

test_path = test_root + "/images"
test_label_path = test_root + "/labels"

save_train_dir = train_root + "/crop_resize/"
save_train_label = train_root + "/crop_label_resize/"

save_test_dir = test_root + "/crop_resize/"
save_test_label = test_root + "/crop_label_resize/"

if not os.path.exists(save_train_dir):
    os.makedirs(save_train_dir)
if not os.path.exists(save_test_dir):
    os.makedirs(save_test_dir)
if not os.path.exists(save_train_label):
    os.makedirs(save_train_label)
if not os.path.exists(save_test_label):
    os.makedirs(save_test_label)

label_prefix = "_labels"

scale_percent = 0.5
brightness_factor = 1.2

crop_params = {
    "train":[40,70,120,60],
    "test":[40,70,150,60]
}

train_images = [os.path.splitext(file)[0] for file in os.listdir(train_path)]
train_labels = [os.path.splitext(file)[0] for file in os.listdir(train_label_path)]

test_images = [os.path.splitext(file)[0] for file in os.listdir(test_path)]
test_labels = [os.path.splitext(file)[0] for file in os.listdir(test_label_path)]

#print(images)
#print(labels)

for (i,image) in enumerate(train_images+test_images):
    print(image)
    number = int(image.split('_')[-1])
    name = '_'.join(image.split('_')[:-1])

    if(image in train_images):
        img_file = glob(train_path + '/'+ image + '.*')[0].replace('\\', '/')
        lf = glob(train_label_path + '/'+ image + label_prefix + '.*')
        if(len(lf)!=0):
            label_file = lf[0].replace('\\', '/')
        else:
            label_file = None
        params = crop_params["train"]
    elif(image in test_images):
        img_file = glob(test_path + '/'+ image + '.*')[0].replace('\\', '/')
        lf = glob(test_label_path + '/'+ image + label_prefix + '.*')
        if(len(lf)!=0):
            label_file = lf[0].replace('\\', '/')
        else:
            label_file = None
        params = crop_params["test"]


    img = cv.imread(img_file, -1)
    if(label_file != None):
        label = cv.imread(label_file, -1)
    else:
        label = np.zeros((img.shape[0], img.shape[1]), np.float32)
    print(label.shape)
    x, y = img.shape
    dim = (int(x*scale_percent), int(y*scale_percent))

    # ig1 = Image.fromarray(label).convert("LA")
    # print(ig1.size)
    # ig1.show()

    # print("before resize:")
    # print(np.max(label), np.max(img))
    # print(np.mean(label), np.mean(img))
    resize_image = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    resize_label = cv.resize(label, dim, interpolation=cv.INTER_CUBIC)*brightness_factor
    crop_image = resize_image[params[0]:resize_image.shape[0]-params[1],params[2]:resize_image.shape[1]-params[3]]
    crop_label = resize_label[params[0]:resize_label.shape[0]-params[1],params[2]:resize_label.shape[1]-params[3]]
    # print("after resize:")
    # print(np.max(resize_label), np.max(resize_image))
    # print(np.mean(resize_label), np.mean(resize_image))
    if(image in train_images):
        cv.imwrite(save_train_dir+f"{image}.png", crop_image)
        cv.imwrite(save_train_label+f"{image}{label_prefix}.png", crop_label)
    elif(image in test_images):
        cv.imwrite(save_test_dir+f"{image}.png", crop_image)
        cv.imwrite(save_test_label+f"{image}{label_prefix}.png", crop_label)
        



# resize_image = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
#     resize_label = cv.resize(label, dim, interpolation=cv.INTER_LINEAR)
#     crop_image =resize_image[50:resize_image.shape[0]-80,120:resize_image.shape[1]-70]
#     crop_label = resize_label[50:resize_label.shape[0]-80,120:resize_label.shape[1]-70]

#     print("before ", np.mean(crop_label), np.max(crop_label))

#     crop_image =crop_image*brightness_factor
#     crop_label = crop_label*brightness_factor