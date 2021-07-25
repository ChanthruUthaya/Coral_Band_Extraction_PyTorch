import os
import cv2 as cv
from glob import glob
import math
import re


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

dir = "D:/2D-remake/3ddata/chunk1"
#dir = os.readlink('scratch')

train_root = dir + '/train_images'
test_root = dir + '/test_images'


image_path = train_root + "/images/"
label_path = train_root+ "/labels/"

test_path = test_root + "/images/"
test_label_path = test_root+ "/labels/"

save_test_dir = dir + "/test_new2/"
save_val_dir = dir + '/vals_new2/'
save_dir = dir + "/train_new2/"


image_dir = f"{save_dir}/images"
label_dir = f"{save_dir}/labels"
test_dir = f"{save_test_dir}/images"
label_test_dir = f"{save_test_dir}/labels"
val_dir = f"{save_val_dir}/images"
label_val_dir = f"{save_val_dir}/labels"

# save_dir = os.readlink('scratch') + "/train"
# save_test_dir = os.readlink('scratch') + "/test"



print(save_dir)

label_prefix = "_labels"

size = 256
stride = 100

max_dim = 22
excluded = []

images_train = [os.path.splitext(file)[0] for file in os.listdir(image_path)]
images_test = [os.path.splitext(file)[0] for file in os.listdir(test_path)]
images_val = [(0,0,0),(0,2,0),(0,3,0),(1,0,0),(2,0,0),(2,1,0),(3,3,1),(3,4,0),(5,1,0),(6,0,0),(6,8,0),(7,1,0),(7,5,1),(9,2,1),(9,8,1)]

#mid = len(images)//2 -1
#images = [images[0],images[-1],images[mid]] 
#test = [os.path.splitext(file)[0] for file in os.listdir(image_path) if not_in(os.path.splitext(file)[0].split("_")[-1])]


#exclude columns
# file_read = open('exclude.txt', 'r')
# lines = file_read.readlines()
# for line in lines:
#     line = line.replace('\n', "").strip()
#     first, second = line.split("-")
#     r_1 = tuple(map(int, first.split(",")))
#     r_2 = tuple(map(int, second.split(",")))
#     #print(r_1, r_2)
#     mult_r_1 = r_1[0]
#     mult_r_2 = r_2[0]
#     x = list(range(mult_r_1*max_dim+r_1[1], mult_r_2*max_dim+r_2[1]+1))
#     out = [(i//max_dim, i%max_dim) for i in x]
#     excluded = excluded + out

# print(images)
#print(labels)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(label_val_dir):
    os.makedirs(label_val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
if not os.path.exists(label_test_dir):
    os.makedirs(label_test_dir)


for (i,image_name) in enumerate(images_train+images_test):
    print(image_name)
    number = int(image_name.split('_')[-1])
    name = '_'.join(image_name.split('_')[:-1])
    #print(name, number)
    # if not os.path.exists(dir_to_save):
    #     os.makedirs(dir_to_save)
    if (image_name in images_train):
        img_file = glob(image_path + '/'+ image_name + '.*')[0].replace('\\', '/')
        label_file = glob(label_path + '/'+ image_name + label_prefix + '.*')[0].replace('\\', '/')
    elif (image_name in images_test):
        img_file = glob(test_path + '/'+ image_name + '.*')[0].replace('\\', '/')
        label_file = glob(test_label_path + '/'+ image_name + label_prefix + '.*')[0].replace('\\', '/')
    image = cv.imread(img_file, -1)
    label = cv.imread(label_file,0)
    bottom_x, bottom_y = image.shape
    print(bottom_x, bottom_y)
    number_x = (bottom_x - size)/stride
    number_y = (bottom_y - size)/stride
    print(f'{i}/{len(images_train+images_test)}')
    print(f"{math.floor(number_x)*math.floor(number_y)} images created")
    #if(image_name in images_train + images_test):
    for x in range(math.floor(number_x)):
        for y in range(math.floor(number_y)):
            x1 = x * stride
            y1 = y * stride
            cropped_image = image[y1:y1+size, x1:x1+size]
            cropped_label = label[y1:y1+size, x1:x1+size]
            i = 0
            if(image_name in images_train):
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label)
                i += 1
                cropped_image_train = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
                cropped_label_train = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image_train)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label_train)
                i = 0
            if(image_name in images_test): #save test
                cv.imwrite(test_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image)
                cv.imwrite(label_test_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label)
                if ((x,y,i) in images_val):
                    cv.imwrite(val_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image)
                    cv.imwrite(label_val_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label)
                i += 1
                cropped_image_test = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
                cropped_label_test = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
                cv.imwrite(test_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image_test)
                cv.imwrite(label_test_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label_test)
                if ((x,y,i) in images_val):
                    cv.imwrite(val_dir + f"/{name}-{x}-{y}-{number}-{i}.png", cropped_image)
                    cv.imwrite(label_val_dir + f"/{name}-{x}-{y}-{number}-{i}-label.png", cropped_label)
                i = 0

