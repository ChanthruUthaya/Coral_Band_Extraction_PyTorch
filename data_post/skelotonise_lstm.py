import argparse
import numpy as np
import math
from model_ablated import *
from sensorAblated import *
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
from pathlib import Path
from sensorAblated import *

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=str, default=".scratch/checkpoint/checkpoint_lstm-0")
parser.add_argument("--dir", type=str, default="./scratch/test_new")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def skeletonise(data_dir, weights):

    #model = SensorAblated(1,1).to(DEVICE)
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

    # data = CoralDataset3DNew(data_dir, mode=1, k=1)
    # #data = CoralDataset(data_dir, augmentations= [], mode=1)
    # data_loader = DataLoader(data, shuffle=False, batch_size=1, num_workers=args.worker_count, pin_memory=True)

    # model.eval()

    # criterion = nn.BCEWithLogitsLoss()

    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine)

    #dir_test = "D:/2D-remake/3ddata/chunk1/test/"
    dir_test = "./scratch/test_new/"
   # test_label = args.dir + "/test"

    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = TransformNew(flips, brightness, affine, mode='3D')

    dataset= CoralDataset3DTest(data_dir,transform, 1, k=3, excluded=['1620','1621'],direction = -1)


    # data = CoralDataset(data_dir, augmentations= [], mode=1)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=args.worker_count, pin_memory=True)


    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            image = batch['image']
            labels = batch['label']
            name = batch['name']
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(image)
            loss = criterion(logits[-1].squeeze(), labels.squeeze())
            logits = torch.sigmoid(logits)
            print(f'[{i}/{len(data_loader)}] batch at loss: {loss.item()}')
            save_skel("./predictions/skel","./predictions/preds",logits.squeeze().cpu().numpy(),name ,i)

def save_skel(save_path_skel, save_path, images, name,i):

        for j in range(images.shape[0]):
            image = images[j]
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

            cv.imwrite(os.path.join(save_path_skel, f"{name}_{j}_skeleton.png"), skel)
            cv.imwrite(os.path.join(save_path, f"{name}_{j}_image.png"), image)

if __name__ == "__main__":
    skeletonise(args.dir, args.resume_checkpoint)
