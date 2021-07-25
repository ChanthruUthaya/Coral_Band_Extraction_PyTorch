import argparse
import numpy as np
import math
from model_ablated import *
from data import *
from transformClass import *
import os
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from ctypes import CDLL
import ctypes
from pathlib import Path

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

"""
1. Take test data, feed through network
2. Take Groundtruths
3. Skeltonise etc...
4. find location of all white pixels in pred, and grountruths
5.go through all pred whites, find location to nearest white pixel
6. go through all groundtruths whites, find location to nearest white pixel in preds
"""

parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=str, default="./scratch/checkpoint_unet/checkpoint")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--dir", type=str, default="./data")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

ret = os.system("cc -fPIC -shared -std=c99 -o accuracy.so accuracy.c -lm")

if ret == 0:
    print("Successfully compiled C library.")
    C = CDLL(os.path.abspath("accuracy.so"))
else:
    print("Couldn't compile C library. Exiting...")
    exit()


def validation_accuracy(outputs, labels):

    test_accuracies = np.zeros(outputs.shape[0])

    for i in range(outputs.shape[0]):

        image = outputs[i]
        label = labels[i]

        image *= 255
        image = np.squeeze(image)
        image = image.astype(np.uint8)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

        # Extract the 2D label, multiply it by 255 so that values are
        # now in the range of 0-255, and threshold.
        label *= 255
        label = label.astype(np.uint8)
        _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)

        # Turn all 255s into 1s for the skeletonization.
        image[image == 255] = 1

        # Skeletonize the thresholded prediction and turn it back into
        # a range of 0-255.
        skel = morphology.skeletonize(image)
        skel = skel.astype(int) * 255

        image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
        label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

        # If a black image is produced then error would be inf. Place
        # a single white pixel to get a finite accuracy score.
        if (len(image_boundaries) == 0):
            image_boundaries = [0, 0]
        

        c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
        c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)
        
        C.euclidean.restype = ctypes.c_double

        test_accuracies[i] = C.euclidean(c_image_boundaries, len(image_boundaries),
                                c_label_boundaries, len(label_boundaries), 5)

    return np.mean(test_accuracies)





def calculate_accuracy(data_dir,weights, tests):

    model = UNetAblated(1,1).to(DEVICE)

     ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(weights, map_location=torch.device('cuda'))
        else:
            # if CPU is used
            checkpoint = torch.load(weights, map_location=torch.device('cpu'))

        print(f"Testing model {weights} that achieved {checkpoint['loss']} loss")

    model.load_state_dict(checkpoint['model'])

    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine)

    #data = CoralDatasetTransfer(data_dir,transform,1)
    data = CoralDataset(data_dir, augmentations= [], mode=1)
    data_loader = DataLoader(data, shuffle=False, batch_size=1, num_workers=args.worker_count, pin_memory=True)

    results = []
    labels_arr = []

    test_accuracies = np.zeros(tests)

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            image = batch['image']
            labels = batch['label']
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(image)
            loss = criterion(logits.squeeze(), labels.squeeze())
            logits = torch.sigmoid(logits)
            print(f'[{i}/{len(data_loader)}] batch at loss: {loss.item()}')
            results.append(logits.cpu().numpy())
            labels_arr.append(labels.cpu().numpy())


    for i, item in enumerate(results):
        print(f'eval {i}/{len(results)}')
        # Extract the 2D prediction, multiply it by 255 so that values are
        # now in the range of 0-255, and threshold using Otsu's method.
        image = item[0, :, :]
        image *= 255
        image = np.squeeze(image)
        image = image.astype(np.uint8)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)

        # Extract the 2D label, multiply it by 255 so that values are
        # now in the range of 0-255, and threshold.
        label = labels_arr[i].squeeze()
        label *= 255
        label = label.astype(np.uint8)
        _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)
        

        # Turn all 255s into 1s for the skeletonization.
        image[image == 255] = 1

        # Skeletonize the thresholded prediction and turn it back into
        # a range of 0-255.
        skel = morphology.skeletonize(image)
        skel = skel.astype(int) * 255

        image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
        label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

        print(len(image_boundaries))
        print(len(label_boundaries))

        # If a black image is produced then error would be inf. Place
        # a single white pixel to get a finite accuracy score.
        if (len(image_boundaries) == 0):
            image_boundaries = [0, 0]
        

        c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
        c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)
        
        C.euclidean.restype = ctypes.c_double


        # Call the C euclidean() method.
        test_accuracies[i] = C.euclidean(c_image_boundaries, len(image_boundaries),
                                c_label_boundaries, len(label_boundaries), 5)

        
        print(f'accuracy of {test_accuracies[i]}')
    
    return np.mean(test_accuracies)

if __name__ == '__main__':
    # accuracies = np.zeros(50)
    # for i in range(60):
    #accuracies[i] = calculate_accuracy(args.dir+"/test/", args.resume_checkpoint+f"-{i}", 56)#
    cp_arg = args.resume_checkpoint+f"-{20}"
    #cp = "./scratch/checkpoint/checkpoint_transfer-4"
    cp = "checkpoint/good_cp/checkpoint-43_focal_good_one"
    data_dir = "data/test"
    accuracies = calculate_accuracy(data_dir,cp,56)
    # print(accuracies[i])
    #  print((90 - accuracies[i]) / 90 * 100)
    print(accuracies)
    # print(list((90 - accuracies) / 90 * 100))

