import argparse
import numpy as np
import math
from model_ablated import *
from data import *
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
from sensorAblated import *
"""
1. Take test data, feed through network
2. Take Groundtruths
3. Skeltonise etc...
4. find location of all white pixels in pred, and grountruths
5.go through all pred whites, find location to nearest white pixel
6. go through all groundtruths whites, find location to nearest white pixel in preds
"""

parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=str, default="./scratch/checkpoint/checkpoint_lstm")
parser.add_argument("--dir", type=str, default="./scratch")
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


def calculate_accuracy(data_dir,weights, tests):

    model = SensorAblatedTest(1,1).to(DEVICE)

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

    transform = TransformNew(flips, brightness, affine, mode='3D')

    dataset= CoralDataset3D(data_dir,transform, 0, k=5, excluded=['1620','1621','1622','1623'], direction=-1)


    # data = CoralDataset(data_dir, augmentations= [], mode=1)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=args.worker_count, pin_memory=True)

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
            loss = criterion(logits[-1,...].squeeze(), labels.squeeze())
            logits = torch.sigmoid(logits)
            print(f'[{i}/{len(data_loader)}] batch at loss: {loss.item()}')
            results.append(logits[-1,...].cpu().numpy())
            labels_arr.append(labels.squeeze().cpu().numpy())


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
        label = labels_arr[i]
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

        image_boundaries = list(np.array(list(np.where(skel == 255))).T.flatten())
        label_boundaries = list(np.array(list(np.where(label == 255))).T.flatten())

        # If a black image is produced then error would be inf. Place
        # a single white pixel to get a finite accuracy score.
        if (len(image_boundaries) == 0):
            image_boundaries = [0, 0]
        
        c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
        c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)
        
        C.euclidean.restype = ctypes.c_double
        # Call the C euclidean() method.
        test_accuracies[i] = C.euclidean(c_image_boundaries, len(image_boundaries),
                                c_label_boundaries, len(label_boundaries))
        acc = (90 - test_accuracies[i]) / 90 * 100
        print(f'accuracy of {acc}')
    
    return np.mean(test_accuracies)

def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))

def euclideanDistance(skel_whites, label_whites):
    distances1 = [float('inf')] * (len(skel_whites)//2)
    distances2 = [float('inf')] * (len(label_whites)//2)

    print(len(skel_whites))
    print(len(label_whites))

    sum1 = 0
    for i in range(0, len(skel_whites)//2):
        for j in range(0, len(label_whites)//2):
            distance_val = distance(skel_whites[i*2],skel_whites[i*2+1], label_whites[j*2],label_whites[j*2+1])
            if distance_val < distances1[i]:
                distances1[i] = distance_val
            if distance_val < distances2[j]:
                distances2[j] = distance_val
        sum1 += distances1[i]
    
    sum2 = 0
    for val in distances2:
        sum2 += val
    
    avg1 = sum1 / len(skel_whites)
    avg2 = sum2 / len(label_whites)

    return (avg1 + avg2)/2

if __name__ == '__main__':
    accuracies = np.zeros(30)
    for i in range(30):
        accuracies[i] = calculate_accuracy(args.dir+"/test_new/", args.resume_checkpoint+f'-{i}', 180)
        print((90 - accuracies) / 90 * 100)

   # print(list((90 - accuracies) / 90 * 100))

