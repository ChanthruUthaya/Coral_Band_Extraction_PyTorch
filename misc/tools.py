import torch
import numpy as np
#import cv2
from matplotlib import pyplot as plt


class EarlyStopping:

    def __init__(self, delta, patience):
        self.delta = delta
        self.patience = patience
        self.previous = None
        self.diff = None
        self.best_Score = None

    def __call__(self, val_loss):
        if self.previous is None and self.diff is None:
            self.diff = val_loss
            self.previous = val_loss
        else:
            self.diff = self.previous - val_loss
            self.previous = val_loss

class ModelCheckpoint:

    def __init__(self, args):
        self.frequency = args.checkpoint_fq
        self.path = args.checkpoint_path
        self.epochs = args.epochs
        self.args = args
    
    def __call__(self, model, loss, epoch):
        ### CHECKPOINT - save parameters, args, accuracy ###
            #Save every args.checkpoint_frequency or if this is the last epoch
        save_name = f"{self.path}-{epoch}"
        if (epoch + 1) % self.frequency == 0 or (epoch + 1) == self.epochs:
            print(f"Saving model to {self.path}-{epoch}")
            torch.save({
                'args': self.args,
                'model': model.state_dict(),
                'loss': loss
            }, save_name)

class NewModelCheckpoint:

    def __init__(self, args):
        self.frequency = args.checkpoint_fq
        self.path = args.checkpoint_path
        self.epochs = args.epochs
        self.args = args
    
    def __call__(self, model, loss, step,epoch):
        ### CHECKPOINT - save parameters, args, accuracy ###
            #Save every args.checkpoint_frequency or if this is the last epoch
        save_name = f"{self.path}-{epoch}"
        if (step) % self.frequency == 0:
            print(f"Saving model to {self.path}-{epoch}")
            torch.save({
                'args': self.args,
                'model': model.state_dict(),
                'loss': loss
            }, save_name)


# def calculate_histogram(image):
#     #hist = cv2.calcHist([image],[0],None,[2],[0,2])
#     hist, bins = np.histogram(image, 2)
#     return hist[0]/hist[1]

# def adjust_data(image):
#         if np.max(image) > 1:
#             image = image / 255
#             image[image > 0.5] = 1
#             image[image <= 0.5] = 0
#         return image

# if __name__ == '__main__':
#     image = cv2.imread("data/train/label/RS0116_0414_0_0_0_label.png",0)
#     image = adjust_data(image).flatten()
#     # print(type(image))
#     print(calculate_histogram(image))
    