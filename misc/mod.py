import cv2 as cv
from data import *
from transformClass import *
from torch.utils.data import random_split

class Modify:
    def __init__(self):
        self.counter = 5

    def modify(self, x):
        x += self.counter
        return x


class Obj1:
    def __init__(self, update):
        self.update = update
    
    def __call__(self, x):
        self.update(x)

class Obj2:
    def __init__(self, update):
        self.update = update
    
    def __call__(self):
        print(self.update.x)

class Update:
    def __init__(self):
        self.x = 0
    
    def __call__(self, x):
        self.x = x


if __name__ == '__main__':
    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine, mode='3D')

    dataset = CoralDataset3D("D:/2D-remake/3ddata/chunk1/train",transform,0, k=1)

    #print(dataset.ids[0])

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    print(len(train),len(val))

    item = train[10]
    image, label = item['image'], item['label']
    print(image.size(), label.size())

