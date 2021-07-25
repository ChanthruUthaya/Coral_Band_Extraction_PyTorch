import torchvision.transforms.functional as TF
import random
from torchvision import transforms
import PIL
from PIL import ImageEnhance
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

class Rotation:

    def __init__(self, angle_range:tuple):
        self.l = angle_range[0] *100
        self.r = angle_range[1] *100

    def __call__(self, image, label):
        angle = random.randrange(self.l, self.r)/100
        return (TF.rotate(image, angle), TF.rotate(label, angle)) 


class Flip:

    def __init__(self, h_flip, v_flip):
        self.h_flip_prob = h_flip
        self.v_flip_prob = v_flip
    
    def __call__(self, image, label, h_flip, v_flip):
        image, label = (TF.hflip(image),TF.hflip(label)) if h_flip else (image,label)
        image, label = (TF.vflip(image),TF.vflip(label)) if v_flip else (image,label)

        return image, label

class AdjustBrightness:

    def __init__(self, brightness_range):
        self.l = brightness_range[0]
        self.r = brightness_range[1]
        self.range = brightness_range

    def __call__(self, image, label):
        image = np.expand_dims(np.array(image).astype(np.float32), axis=2)
        #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        #print(np.max(image))
       # print(image.shape)
       # image = cv2.convertScaleAbs(np.array(image).astype(np.uint16))
       # val = random.uniform(self.l,self.r)
        image = tf.keras.preprocessing.image.random_brightness(image, self.range)
        image = image.squeeze()
        #print(np.max(image))
        #image = np.clip(image*val, 0.0, 255.0)
        return image, np.array(label).astype(np.float32)

class Affine:
    def __init__(self, translate:tuple, shear_range:tuple, scale:float, angle:tuple):
        self.translate = translate
        self.shear_range = shear_range
        self.scale = scale
        self.angle = angle
    def __call__(self, image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val):
        image, label = (TF.affine(image,
                                    angle = angle, 
                                    translate = (h_trans_val, v_trans_val),
                                    scale = scale_val, shear = shear_val), 
                        TF.affine(label, 
                                    angle = angle, 
                                    translate = (h_trans_val, v_trans_val), 
                                    scale = scale_val, 
                                    shear = shear_val)) 
       

        return image, label

class Transform:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine, mode='2D'):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
        self.mode = mode

    def gen_flip_args(self):
        h_flip = True if random.random() < self.flip.h_flip_prob else False
        v_flip = True if random.random() < self.flip.v_flip_prob else False
        return h_flip, v_flip

    
    def gen_affine_args(self):
        scale_val = random.uniform(1-self.affine.scale, 1+self.affine.scale)
        shear_val = random.uniform(self.affine.shear_range[0], self.affine.shear_range[1])
        angle = random.uniform(self.affine.angle[0], self.affine.angle[1])
        h_trans_val = random.uniform(self.affine.translate[0][0], self.affine.translate[0][1])
        v_trans_val = random.uniform(self.affine.translate[1][0], self.affine.translate[1][1])

        return scale_val, shear_val, angle, h_trans_val, v_trans_val
    
    def __call__(self, image, label):

        if(self.mode == '2D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()

            image, label = self.flip(image, label, h_flip, v_flip)
            image, label = self.affine(image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
            # image, label = self.brightness(image, label) ##COMMENTED OUT FOR TRANSFER

            return image, label
        
        elif(self.mode =='3D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()
            res_img = []
            res_label = []
            for img, img_label in zip(image, label):
                if(str(type(img).__module__) == 'PIL.TiffImagePlugin'):
                    img, img_label = self.flip(img, img_label, h_flip, v_flip)
                    img, img_label = self.affine(img, img_label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
                    #img, img_label = self.brightness(img, img_label)

                res_img.append(img)
                res_label.append(img_label)

            return res_img, res_label

class Transform2D:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine, mode='2D'):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
        self.mode = mode

    def gen_flip_args(self):
        h_flip = True if random.random() < self.flip.h_flip_prob else False
        v_flip = True if random.random() < self.flip.v_flip_prob else False
        return h_flip, v_flip

    
    def gen_affine_args(self):
        scale_val = random.uniform(1-self.affine.scale, 1+self.affine.scale)
        shear_val = random.uniform(self.affine.shear_range[0], self.affine.shear_range[1])
        angle = random.uniform(self.affine.angle[0], self.affine.angle[1])
        h_trans_val = random.uniform(self.affine.translate[0][0], self.affine.translate[0][1])
        v_trans_val = random.uniform(self.affine.translate[1][0], self.affine.translate[1][1])

        return scale_val, shear_val, angle, h_trans_val, v_trans_val
    
    def __call__(self, image, label):

        if(self.mode == '2D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()

            image, label = self.flip(image, label, h_flip, v_flip)
            image, label = self.affine(image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
            image, label = self.brightness(image, label) ##COMMENTED OUT FOR TRANSFER

            return image, label
        
        elif(self.mode =='3D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()
            res_img = []
            res_label = []
            for img, img_label in zip(image, label):
                if(str(type(img).__module__) == 'PIL.TiffImagePlugin'):
                    img, img_label = self.flip(img, img_label, h_flip, v_flip)
                    img, img_label = self.affine(img, img_label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
                    #img, img_label = self.brightness(img, img_label)

                res_img.append(img)
                res_label.append(img_label)

            return res_img, res_label


class TransformNew:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine, mode='2D'):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
        self.mode = mode

    def gen_flip_args(self):
        h_flip = True if random.random() < self.flip.h_flip_prob else False
        v_flip = True if random.random() < self.flip.v_flip_prob else False
        return h_flip, v_flip

    
    def gen_affine_args(self):
        scale_val = random.uniform(1-self.affine.scale, 1+self.affine.scale)
        shear_val = random.uniform(self.affine.shear_range[0], self.affine.shear_range[1])
        angle = random.uniform(self.affine.angle[0], self.affine.angle[1])
        h_trans_val = random.uniform(self.affine.translate[0][0], self.affine.translate[0][1])
        v_trans_val = random.uniform(self.affine.translate[1][0], self.affine.translate[1][1])

        return scale_val, shear_val, angle, h_trans_val, v_trans_val
    
    def __call__(self, image, label):

        if(self.mode == '2D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()

            image, label = self.flip(image, label, h_flip, v_flip)
            image, label = self.affine(image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
            image, label = self.brightness(image, label)

            return image, label
        
        elif(self.mode =='3D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()
            res_img = []
            res_label = []
            for img, img_label in zip(image, label):
                # if(str(type(img).__module__) == 'PIL.TiffImagePlugin'):
                img, img_label = self.flip(img, img_label, h_flip, v_flip)
                img, img_label = self.affine(img, img_label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
                img, img_label = self.brightness(img, img_label)

                res_img.append(img)
                res_label.append(img_label)

            return res_img, res_label



