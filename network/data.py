from glob import glob
import numpy as np
import os
import skimage.io as io
# import skimage.transform as trans
from skimage import img_as_ubyte, morphology
import torch
from transformClass import *
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import random
from tensorflow.python.keras import backend


import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
# from tifffile import imread

def not_in(string, excluded):
    if string not in excluded:
        return True
    else:
        return False


# class CoralDataset(Dataset):
#     def __init__(self, img_dir, label_dir,augmentations, mode, label_suffix = '_label'):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.label_suffix = label_suffix
#         self.augmentations = augmentations
#         self.mode = mode
#         self.ids = [os.path.splitext(file)[0] for file in os.listdir(img_dir)]
    
#     def __len__(self):
#         return len(self.ids)
    
#     @staticmethod
#     def adjust_data(image, label):
#         if np.max(image) > 1:
#             image = image / 255
#             label = label / 255
#             label[label > 0.5] = 1
#             label[label <= 0.5] = 0

#         return Image.fromarray(image), Image.fromarray(label)

#     def augment(self, img, label):
#         img, label = CoralDataset.adjust_data(img, label)
#         #img, label = Image.fromarray(img), Image.fromarray(label)
#         trans = transforms.Compose([
#                 transforms.ToTensor()
#                 ])
#         if(self.mode == 0):
#             img, label = self.augmentations(img, label)
#         return trans(img), trans(label)

#     def __str__(self):
#         return '[' + ' , '.join(self.ids) + ']'

#     def __getitem__(self, i):

#         indexes = [j for j in range(len(self.ids)) if j != i]
#         afteridx = random.choice(indexes)
#         indexes.remove(afteridx)
#         beforeidx = random.choice(indexes)

#         idx = [self.ids[beforeidx], self.ids[i], self.ids[afteridx]]

#         label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
#         img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

#         img_0 = Image.open(img_file[1].replace('\\', '/'))
#         label_0 = Image.open(label_file[1].replace('\\', '/'))

#         labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
#         images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

#         # zips = list(map(self.augment, images, labels))
#         # img, label, weights = torch.stack([i for i,_,_ in zips]), torch.stack([j for _, j,_ in zips]), torch.stack([k for _,_,k in zips])

#         # if i == 2:
#         #     img_0.show()
#         #     label_0.show()

#         img_0, label_0 = self.augment(np.array(img_0), np.array(label_0))

#         # if i == 2:
#         #     img_0.show()
#         #     label_0.show()


#         # return {
#         #     'image': img_0,
#         #     'label': label_0
#         # }
#         return {
#             'image': img_0,
#             'label': label_0,
#         }

class CoralDataset2D(Dataset):
    """2D Coral slices dataset."""

    def __init__(self, sample_dir, label_dir, transform=None):
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(glob(f"{self.sample_dir}/*.png"))

    def __getitem__(self, idx):
        f = sorted(glob(f"{self.sample_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        sample = io.imread(name)
        sample = transforms.functional.to_pil_image(sample)

        f = sorted(glob(f"{self.label_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        label = io.imread(name)
        threshold = label < 0.5
        label[threshold] = 0
        label = transforms.functional.to_pil_image(label)

        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)

        return sample, label


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


        # for image, label in self.train_generator:
        #     print("adding")
        #     image, label = CoralDataset.adjust_data(image, label)
        #     self.data.append((image,label))

    def generator(self):
        image_datagen = ImageDataGenerator(**self.aug_dict)
        label_datagen = ImageDataGenerator(**self.aug_dict)

        seed = np.random.randint(0, 100)

    # The same seed argument is used when the image and label generators are
    # created to ensure that the same transformations are applied to both.
        image_generator = image_datagen.flow_from_directory(
            self.dir,
            classes=["image"],
            class_mode=None,
            color_mode="grayscale",
            target_size=(256,256),
            batch_size=2,
            save_to_dir=None,
            save_prefix="image",
            seed=seed
        )

        label_generator = label_datagen.flow_from_directory(
            self.dir,
            classes=["label"],
            class_mode=None,
            color_mode="grayscale",
            target_size=(256,256),
            batch_size=2,
            save_to_dir=None,
            save_prefix="label",
            seed=seed
        )

        # Zip the two generators into one.
        train_generator = zip(image_generator, label_generator)

        for image, label in train_generator:
            image, label = CoralDataset.adjust_data(image, label)
            image, label = torch.from_numpy(image), torch.from_numpy(label)
            yield image.view(2,-1,256,256), label.squeeze()

    
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

        indexes = [j for j in range(len(self.ids)) if j != i]
        afteridx = random.choice(indexes)
        indexes.remove(afteridx)
        beforeidx = random.choice(indexes)

        idx = [self.ids[beforeidx], self.ids[i], self.ids[afteridx]]

        #print(self.ids[i])

        # label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
        # img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

        label_file = glob(self.label_dir + '/' + self.ids[i] + self.label_suffix + '.*')[0].replace('\\', '/')
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0].replace('\\', '/')

        img_0 = Image.open(img_file)
        label_0 = Image.open(label_file)


        #img_0.show()

        #print(str(img_file))


        # labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
        # images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

        img_0, label_0 = self.augment(img_0, label_0)
        img_0, label_0 = CoralDataset.adjust_data(np.array(img_0).astype(np.float32), np.array(label_0).astype(np.float32))

        # img, label = self.data[i]
        # img, label - self.augment(img, label)



        # print("img size",img_0.shape)
        # print("label size", label_0.shape)

        return {
            'image': transforms.ToTensor()(img_0),
            'label': transforms.ToTensor()(label_0),
            'name': self.ids[i]
        }#

class CoralDatasetTransfer(Dataset):

    #training old 2d data on old model


    def __init__(self, dir ,augmentations, mode, label_suffix = '-label'):  
        ##for transfer changes _label to -label and /image to /images /label to /labels
        self.dir = dir
        self.img_dir = dir + "images/"
        self.label_dir = dir + "labels/"
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.brightness = AdjustBrightness((0.9,1.1))
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]
   
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            #print(np.max(image))
            image = (image / 65535.0)*255.0
            label = (label / 255.0) #shift brightness from resize
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return (image, label)

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

       #print("before augment ", img_0.getextrema())

        img_0, label_0 = self.augment(img_0, label_0)

        #print("After augment ", img_0.getextrema())


        img_0, label_0 = CoralDataset.adjust_data(np.array(img_0).astype(np.float32), np.array(label_0).astype(np.float32))

        #print("After adjust ", np.max(img_0) )

        if(self.mode == 0):
            img_0, label_0 = self.brightness(img_0, label_0)
        
        #print("After bright ", np.max(img_0) )
        img_0 = img_0 /255.0


        return {
            'image': transforms.ToTensor()(img_0),
            'label': transforms.ToTensor()(label_0)
        }

class CoralDatasetTransferTest(Dataset):

    #training old 2d data on old model


    def __init__(self, dir ,augmentations, mode, label_suffix = '-label'):  
        ##for transfer changes _label to -label and /image to /images /label to /labels
        self.dir = dir
        self.img_dir = dir + "images/"
        self.label_dir = dir + "labels/"
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.brightness = AdjustBrightness((0.9,1.1))
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]
   
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            #print(np.max(image))
            image = (image / 65535.0)*255.0
            label = (label / 255.0) #shift brightness from resize
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return (image, label)

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

       #print("before augment ", img_0.getextrema())

        img_0, label_0 = self.augment(img_0, label_0)

        #print("After augment ", img_0.getextrema())


        img_0, label_0 = CoralDataset.adjust_data(np.array(img_0).astype(np.float32), np.array(label_0).astype(np.float32))

        #print("After adjust ", np.max(img_0) )

        if(self.mode == 0):
            img_0, label_0 = self.brightness(img_0, label_0)
        
        #print("After bright ", np.max(img_0) )
        img_0 = img_0 /255.0


        return {
            'image': transforms.ToTensor()(img_0),
            'label': transforms.ToTensor()(label_0),
            'name': self.ids[i]
        }




class CoralDatasetNew(Dataset):

    #training new data for 2d

    def __init__(self, dir ,augmentations, mode, label_suffix = '-label'):
        self.dir = dir
        self.img_dir = dir + "images/cuts_images"
        self.label_dir = dir + "labels/cuts_labels"
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]
    
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
        
        # print(i)
        # print(len(self.ids))
        # #print(self.ids[i])


        # label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
        # img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

        label_file = glob(self.label_dir + '/' + self.ids[i] + self.label_suffix + '.*')[0].replace('\\', '/')
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0].replace('\\', '/')

        img_0 = Image.open(img_file)
        label_0 = Image.open(label_file)


        #img_0.show()

        #print(str(img_file))


        # labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
        # images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

        img_0, label_0 = self.augment(img_0, label_0)
        img_0, label_0 = CoralDataset.adjust_data(np.array(img_0).astype(np.float32), np.array(label_0).astype(np.float32))

        # img, label = self.data[i]
        # img, label - self.augment(img, label)



        # print("img size",img_0.shape)
        # print("label size", label_0.shape)

        return {
            'image': transforms.ToTensor()(img_0),
            'label': transforms.ToTensor()(label_0)
        }

class CoralDataset3D(Dataset):

    #for training new dataset on 3d data

    #changed so that no transform applied

    def __init__(self, dir ,augmentations, mode,k=3,size=256,step=1 ,label_suffix = '-label', excluded =[], direction=1):
        super().__init__()
        self.dir = dir
        self.img_dir = dir + "/images/"
        self.label_dir = dir + "/labels/"
        self.augmentations = augmentations
        self.mode = mode
        self.label_suffix = label_suffix
        self.brightness = AdjustBrightness((0.9,1.1))
        self.k = k
        self.size = size
        self.step = 1
        self.direction = direction
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir) if not_in(os.path.splitext(file)[0].split('-')[-2], excluded)]

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            #print(np.max(image))
            image = (image / 65535.0)*255.0
            label = label / 255.0
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return (image, label)
    
    def augment(self, img, label):
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        return img, label

    @staticmethod
    def get_image_data(name):
        name, x, y, slice_num, rotation  = name.split('-')
        return name, x, y, slice_num, rotation 

    def __getitem__(self, i):
        image_name = self.ids[i]
        name, x, y, slice_num, rotation = CoralDataset3D.get_image_data(image_name)#
      #  print("getting image", image_name)
        num_of_slices = (self.k-1) #//2
        ##slice_nums_to_get = [num for sublist in [[int(slice_num)-i*self.step,int(slice_num)+i*self.step] for i in range(1,num_of_slices+1)] for num in sublist]
        slice_nums_to_get = [int(slice_num)-i*self.direction for i in range(1,num_of_slices+1)]
        slice_nums_to_get.append(int(slice_num))
        if(self.direction == 1):
            slice_nums_to_get.sort()
        else:
            slice_nums_to_get.sort(reverse=True)

        #print(slice_nums_to_get)

        label_files = []
        image_files = []

        for num in slice_nums_to_get:
            name_string = '-'.join([name,x,y,str(num),rotation])
         #   print(f'name string {name_string} stack {image_name}')
            img_file = glob(f'{self.img_dir}/{name_string}.*')
            if(len(img_file)!=0):
                image = Image.open(img_file[0].replace('\\', '/'))
                image_files.append(image)
                label_file = glob(f'{self.label_dir}/{name_string}{self.label_suffix}.*')[0].replace('\\', '/')
                label_files.append(Image.open(label_file.replace('\\', '/')))
            # else:
            #     image_files.append(np.zeros((self.size, self.size), np.float32))
            #     label_files.append(np.zeros((self.size, self.size), np.float32))

        image_files, label_files = self.augment(image_files, label_files)

        adjusted = [CoralDataset3D.adjust_data(np.array(img).astype(np.float32),np.array(label).astype(np.float32)) for img, label in zip(image_files, label_files)]

        if(self.mode == 0):
            adjusted = [self.brightness(img[0], img[1]) for img in adjusted]

        # for img, label in bright_adjust:
        #     img_g = Image.fromarray(img)
        #     label_g = Image.fromarray(label*255)

        #     img_g.show()
        #     label_g.show()
    #    # mid = (len(adjusted)-1)//2
    #     img_pil = Image.fromarray(adjusted[-1][0])
    #     label_pil = Image.fromarray(adjusted[-1][1]*255)
    #     img_pil.show()
    #     label_pil.show()

        images = torch.stack([transforms.ToTensor()(img[0]/255.0) for img in adjusted], axis=0)
        labels = transforms.ToTensor()(adjusted[-1][1])

        

        return {
            'image':images,
            'label':labels
        }



class CoralDataset3DTest(Dataset):

    #for training new dataset on 3d data

    #changed so that no transform applied

    def __init__(self, dir ,augmentations, mode,k=3,size=256,step=1 ,label_suffix = '-label', excluded =[], direction=1):
        super().__init__()
        self.dir = dir
        self.img_dir = dir + "/images/"
        self.label_dir = dir + "/labels/"
        self.augmentations = augmentations
        self.mode = mode
        self.label_suffix = label_suffix
        self.brightness = AdjustBrightness((0.9,1.1))
        self.k = k
        self.size = size
        self.step = 1
        self.direction = direction
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir) if not_in(os.path.splitext(file)[0].split('-')[-2], excluded)]

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            #print(np.max(image))
            image = (image / 65535.0)*255.0
            label = label / 255.0
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return (image, label)
    
    def augment(self, img, label):
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        return img, label

    @staticmethod
    def get_image_data(name):
        name, x, y, slice_num, rotation  = name.split('-')
        return name, x, y, slice_num, rotation 

    def __getitem__(self, i):
        image_name = self.ids[i]
        name, x, y, slice_num, rotation = CoralDataset3D.get_image_data(image_name)#
      #  print("getting image", image_name)
        num_of_slices = (self.k-1) #//2
        ##slice_nums_to_get = [num for sublist in [[int(slice_num)-i*self.step,int(slice_num)+i*self.step] for i in range(1,num_of_slices+1)] for num in sublist]
        slice_nums_to_get = [int(slice_num)-i*self.direction for i in range(1,num_of_slices+1)]
        slice_nums_to_get.append(int(slice_num))
        if(self.direction == 1):
            slice_nums_to_get.sort()
        else:
            slice_nums_to_get.sort(reverse=True)

        #print(slice_nums_to_get)

        label_files = []
        image_files = []

        for num in slice_nums_to_get:
            name_string = '-'.join([name,x,y,str(num),rotation])
         #   print(f'name string {name_string} stack {image_name}')
            img_file = glob(f'{self.img_dir}/{name_string}.*')
            if(len(img_file)!=0):
                image = Image.open(img_file[0].replace('\\', '/'))
                image_files.append(image)
                label_file = glob(f'{self.label_dir}/{name_string}{self.label_suffix}.*')[0].replace('\\', '/')
                label_files.append(Image.open(label_file.replace('\\', '/')))
            # else:
            #     image_files.append(np.zeros((self.size, self.size), np.float32))
            #     label_files.append(np.zeros((self.size, self.size), np.float32))

        image_files, label_files = self.augment(image_files, label_files)

        adjusted = [CoralDataset3D.adjust_data(np.array(img).astype(np.float32),np.array(label).astype(np.float32)) for img, label in zip(image_files, label_files)]

        if(self.mode == 0):
            adjusted = [self.brightness(img[0], img[1]) for img in adjusted]

        # for img, label in bright_adjust:
        #     img_g = Image.fromarray(img)
        #     label_g = Image.fromarray(label*255)

        #     img_g.show()
        #     label_g.show()
    #    # mid = (len(adjusted)-1)//2
    #     img_pil = Image.fromarray(adjusted[-1][0])
    #     label_pil = Image.fromarray(adjusted[-1][1]*255)
    #     img_pil.show()
    #     label_pil.show()

        images = torch.stack([transforms.ToTensor()(img[0]/255.0) for img in adjusted], axis=0)
        labels = transforms.ToTensor()(adjusted[-1][1])
    

        return {
            'image':images,
            'label':labels,
            'name': image_name
        }




class CoralDataset3DBLSTM(Dataset):

    #for training new dataset on 3d data

    #changed so that no transform applied

    def __init__(self, dir ,augmentations, mode,k=3,size=256,step=1 ,label_suffix = '-label', excluded =[]):
        super().__init__()
        self.dir = dir
        self.img_dir = dir + "/images/"
        self.label_dir = dir + "/labels/"
        self.augmentations = augmentations
        self.mode = mode
        self.label_suffix = label_suffix
        self.brightness = AdjustBrightness((0.9,1.1))
        self.k = k
        self.size = size
        self.step = 1
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir) if not_in(os.path.splitext(file)[0].split('-')[-2], excluded)]

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            #print(np.max(image))
            image = (image / 65535.0)*255.0
            label = label / 255.0
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return (image, label)
    
    def augment(self, img, label):
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        return img, label

    @staticmethod
    def get_image_data(name):
        name, x, y, slice_num, rotation  = name.split('-')
        return name, x, y, slice_num, rotation 

    def __getitem__(self, i):
        image_name = self.ids[i]
        name, x, y, slice_num, rotation = CoralDataset3D.get_image_data(image_name)#
      #  print("getting image", image_name)
        num_of_slices = (self.k-1) #//2
        ##slice_nums_to_get = [num for sublist in [[int(slice_num)-i*self.step,int(slice_num)+i*self.step] for i in range(1,num_of_slices+1)] for num in sublist]
        slice_nums_to_get = [num for num_list in [[int(slice_num)-i,int(slice_num)+i] for i in range(1,num_of_slices+1)] for num in num_list]
        slice_nums_to_get.append(int(slice_num))
        slice_nums_to_get.sort()

        #print(slice_nums_to_get)

        label_files = []
        image_files = []

        for num in slice_nums_to_get:
            name_string = '-'.join([name,x,y,str(num),rotation])
         #   print(f'name string {name_string} stack {image_name}')
            img_file = glob(f'{self.img_dir}/{name_string}.*')
            if(len(img_file)!=0):
                image = Image.open(img_file[0].replace('\\', '/'))
                image_files.append(image)
                label_file = glob(f'{self.label_dir}/{name_string}{self.label_suffix}.*')[0].replace('\\', '/')
                label_files.append(Image.open(label_file.replace('\\', '/')))
            else:
                image_files.append(np.zeros((self.size, self.size), np.float32))
                label_files.append(np.zeros((self.size, self.size), np.float32))

        image_files, label_files = self.augment(image_files, label_files)

        adjusted = [CoralDataset3D.adjust_data(np.array(img).astype(np.float32),np.array(label).astype(np.float32)) for img, label in zip(image_files, label_files)]

        if(self.mode == 0):
            adjusted = [self.brightness(img[0], img[1]) for img in adjusted]

        # for img, label in bright_adjust:
        #     img_g = Image.fromarray(img)
        #     label_g = Image.fromarray(label*255)

        #     img_g.show()
        #     label_g.show()
    #    # mid = (len(adjusted)-1)//2
    #     img_pil = Image.fromarray(adjusted[-1][0])
    #     label_pil = Image.fromarray(adjusted[-1][1]*255)
    #     img_pil.show()
    #     label_pil.show()

        ##CHANGED FOR BIDIRECTION
        #MID and slice numstoget also changed to load in black array 

        mid = num_of_slices + 1

        images = torch.stack([transforms.ToTensor()(img[0]/255.0) for img in adjusted], axis=0)
        labels = transforms.ToTensor()(adjusted[mid][1])

        

        return {
            'image':images,
            'label':labels
        }

class CoralDataset3DNew(Dataset):

    def __init__(self, dir,mode,augmentations=None,k=3,size=256,step=1 ,label_suffix = '_label'):
        super().__init__()
        self.dir = dir
        self.img_dir = dir + "/image"
        self.label_dir = dir + "/label"
        self.augmentations = augmentations
        self.mode = mode
        self.label_suffix = label_suffix
        self.k = k
        self.size = size
        self.step = 1
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
          #  print(np.max(image))
            image = image / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return image, label
    
    def augment(self, img, label):
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        # else:
        #     img, label np.array(img), np.array(label)
        return img, label

    @staticmethod
    def get_image_data(name):
        name, x, y, slice_num, rotation  = name.split('-')
        return name, x, y, slice_num, rotation 

    def __getitem__(self, i):
        image_name = self.ids[i]
        # name, x, y, slice_num, rotation = CoralDataset3D.get_image_data(image_name)
        # num_of_slices = (self.k-1)//2
        # slice_nums_to_get = []#[num for sublist in [[int(slice_num)-i*self.step,int(slice_num)+i*self.step] for i in range(1,num_of_slices+1)] for num in sublist]
        # slice_nums_to_get.append(int(slice_num))
        # slice_nums_to_get.sort()
 
        label_files = []
        image_files = []

        label_file = glob(self.label_dir + '/' + self.ids[i] + self.label_suffix + '.*')[0].replace('\\', '/')
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0].replace('\\', '/')

        img_0 = Image.open(img_file)
        label_0 = Image.open(label_file)

        label_files.append(label_0)
        image_files.append(img_0)

        # for num in slice_nums_to_get:
        #     name_string = '-'.join([name,x,y,str(num),rotation])
        #     img_file = glob(f'{self.img_dir}/{name_string}.*')
        #     if(len(img_file)!=0):
        #         image = Image.open(img_file[0].replace('\\', '/'))
        #         image_files.append(image)
        #         #print(image.getextrema())
        #         label_file = glob(f'{self.label_dir}/{name_string}{self.label_suffix}.*')[0].replace('\\', '/')
        #         label_files.append(Image.open(label_file.replace('\\', '/')))
        #     else:
        #         image_files.append(np.zeros((self.size, self.size), np.float32))
        #         label_files.append(np.zeros((self.size, self.size), np.float32))

        image_files, label_files = self.augment(image_files, label_files)

        adjusted = [CoralDataset3DNew.adjust_data(np.array(img).astype(np.float32),np.array(label).astype(np.float32)) for img, label in zip(image_files, label_files)]

        mid = (len(adjusted)-1)//2


        images = torch.stack([transforms.ToTensor()(img[0]) for img in adjusted], axis=0)
        labels = transforms.ToTensor()(adjusted[mid][1])

        return {
            'image':images,
            'label':labels
        }



    





# def save_predictions(save_path, predictions):
#     for i, batch in enumerate(predictions):
#         for j, item in enumerate(batch):

#             image = item[0, :, :]
#            #
#            #  label_img = label[0, :, :]

#             output = img_as_ubyte(image)
#            # output_label = img_as_ubyte(label_img)

#             # Threshold the image using Otsu's method.
#             _, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

#             # Replace all 255s with 1 in preparation for the skeletonization.
#             output[output == 255] = 1

#             # conf = gather_data(output, output_label)
#             # print(conf)
#             # cer, f_val = metrics(conf)
#             # print(f'CER val {cer}, F score {f_val}'



# # Skeletonize the thresholded predictions.
#             skel = morphology.skeletonize(output)
# #             skel = skel.astype(int) * 255

# # #Output the skeletonized prediction.
# #             print("Saving prediction to out.png")

# #             cv.imwrite(os.path.join(save_path, f"{i}{j}_predict.png"), np.array(skel))

# def save_predictions(save_path, image, i):
#     #image = image[0, :, :]
# #
# #  label_img = label[0, :, :]

#     output = img_as_ubyte(image)
# #  # output_label = img_as_ubyte(label_img)

# #   # Threshold the image using Otsu's method.
#     _, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

# #   # Replace all 255s with 1 in preparation for the skeletonization.
#     output[output == 255] = 1

# #   # Skeletonize the thresholded predictions.
#     skel = morphology.skeletonize(output)
#     skel = skel.astype(int) * 255

#     #Output the skeletonized prediction.
#     print("Saving prediction to out.png")

#     print(output.shape)

#     cv.imwrite(os.path.join(save_path, f"{i}_predict.png"), skel)

def save_pred(save_path, image, i):
        #image = image[0, :, :]
#
#  label_img = label[0, :, :]

    output = img_as_ubyte(image)
#  # output_label = img_as_ubyte(label_img)
    #Output the skeletonized prediction.
    print("Saving prediction to out.png")

    print(output.shape)

    cv.imwrite(os.path.join(save_path, f"{i}_predict.png"), output)

def save_pred_lstm(save_path, images, i):

    for j in range(images.shape[0]):

        output = img_as_ubyte(images[j])
#  # output_label = img_as_ubyte(label_img)
    #Output the skeletonized prediction.
        print("Saving prediction to out.png")

        print(output.shape)

        cv.imwrite(os.path.join(save_path, f"{i}_{j}_predict.png"), output)


# # train_dataset = datasets.ImageFolder(
# #     traindir,
# #     transforms.Compose([
# #         transforms.RandomResizedCrop(224),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.ToTensor(),
# #         normalize,
# #     ]))
# # train_loader = torch.utils.data.DataLoader(
# #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
# #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)


# # torchvision.transforms.functional.affine -- shear
# # torchvision.transforms.RandomAffine -- translate (shift H + W)
# # torchvision.transforms.RandomAffine -- zoom ()
# #fill_mode = nearest