from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2 as cv
from skimage import img_as_ubyte, morphology



class CoralDataset(Dataset):

    #training old 2d data on old model


    def __init__(self, dir ,augmentations, mode, label_suffix = '_label', aug_dict=dict()):  
        ##for transfer changes _label to -label and /image to /images /label to /labels
        self.dir = dir
        self.img_dir = dir + "/image/"
        self.label_dir = dir + "/label/"
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]
        self.aug_dict = aug_dict


    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            image = image / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return image, label

    def augment(self, img, label):
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        return img, label

    def __str__(self):
        return '[' + ' , '.join(self.ids) + ']'

    def __getitem__(self, i):

        label_file = glob(self.label_dir + '/' + self.ids[i] + self.label_suffix + '.*')[0].replace('\\', '/')
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0].replace('\\', '/')

        img_0 = Image.open(img_file)
        label_0 = Image.open(label_file)

        img_0, label_0 = self.augment(img_0, label_0)
        img_0, label_0 = CoralDataset.adjust_data(np.array(img_0).astype(np.float32), np.array(label_0).astype(np.float32))


        return {
            'image': transforms.ToTensor()(img_0),
            'label': transforms.ToTensor()(label_0),
            'name': self.ids[i]
        }

def save_skel(save_path_skel, save_path, image, i):

        image *= 255
        image = image.astype(np.uint8)
        _, image_threh = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

        # Turn all 255s into 1s for the skeletonization.
        image_threh[image_threh == 255] = 1

        # Skeletonize the thresholded prediction and turn it back into
        # a range of 0-255.
        skel = morphology.skeletonize(image_threh)
        skel = skel.astype(int) * 255

    #  # output_label = img_as_ubyte(label_img)
        #Output the skeletonized prediction.
        print("Saving prediction")

        cv.imwrite(os.path.join(save_path_skel, f"{i}_skeleton.png"), skel)
        cv.imwrite(os.path.join(save_path, f"{i}_image.png"), image)