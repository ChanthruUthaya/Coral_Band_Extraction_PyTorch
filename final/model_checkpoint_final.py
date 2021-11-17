import torch
import numpy as np
#import cv2
from matplotlib import pyplot as plt

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
                'model': model.state_dict(),
                'loss': loss
            }, save_name)