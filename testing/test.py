
import argparse
from data import *
from models import *
#from models2 import *
from model_ablated import *
from models_test import *
from tools import *
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

import time

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=str, default="./checkpoint/checkpoint")
parser.add_argument("--batch-size", default=2, type=int, help="Number of images within each mini-batch")
parser.add_argument("--dir", type=str, default="./data")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    model = UNetAblated(1,1).to(DEVICE)


    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cuda'))
        else:
            # if CPU is used
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

        print(f"Testing model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")

        model.load_state_dict(checkpoint['model'])

    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine)

    dir_test = "scratch/test_new/"
   # test_label = args.dir + "/test"

    test_data= CoralDatasetTransfer(dir_test,transform, mode=1)
    #test_data = CoralDataset(dir_test, augmentations=[] ,mode=1)
    test_loader = DataLoader(test_data, shuffle=False ,batch_size=1, num_workers=args.worker_count, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    model.eval()
    preds = []

        # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch['image']
            labels = batch['label']
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(image)
            loss = criterion(logits.squeeze(), labels.squeeze())
            logits = torch.sigmoid(logits)
            print(f'[{i}/{len(test_loader)}] batch at loss: {loss.item()}')
            total_loss += loss.item()
            preds.append(logits.cpu().numpy())
            print(np.max(logits.squeeze().cpu().numpy()))
            save_pred("./predictions",logits.squeeze().cpu().numpy(), i)

    
    average_loss = total_loss / len(test_loader)

    print(average_loss)


if __name__ == '__main__':
    main(parser.parse_args())