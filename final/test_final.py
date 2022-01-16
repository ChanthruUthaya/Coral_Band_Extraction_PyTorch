
#Library Imports
import os
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from pathlib import Path
from types import SimpleNamespace
import statistics as stats
import argparse
import time
import skimage.io as io
from skimage import img_as_ubyte, morphology
from PIL import Image
import random
from tensorflow.python.keras import backend
from glob import glob
import cv2 as cv
import time

#File Imports

from data_final import *
from loss_functions_final import *
from model_final import *
from model_checkpoint_final import *
from transformClass_final import *

parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=str, default = "../../checkpoint/checkpoint_folder/")
parser.add_argument("--checkpoint", type=str, default = "checkpoint-24")
parser.add_argument("-j", "--worker-count", default = cpu_count(), type=int, help="Number of worker processes used to load data")
parser.add_argument("--data-dir", type=str, default="../../data/")
parser.add_argument("--preds", type=str, default="predictions")
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--batch-size", type=int, default=1)

args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def test_main(args):
 
  model = UNetAblated(1,1)

  checkpoint_path = args.resume_checkpoint + args.checkpoint
  save_path = args.data_dir + args.preds

  if args.resume_checkpoint != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        else:
            # if CPU is used
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        print(f"Testing model {checkpoint_path} that achieved {checkpoint['loss']} loss")

        model.load_state_dict(checkpoint['model'])

  flips = Flip(0.5, 0.5)
  brightness = AdjustBrightness((0.9,1.1))
  affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

  transform = Transform(flips, brightness, affine)

  test_data_dir = args.data_dir + args.mode

  test_data = CoralDataset(test_data_dir, transform, mode=1)
  test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.worker_count, pin_memory=True)

  criterion = nn.BCEWithLogitsLoss()

  total_loss = 0
  model.eval()
  preds = []

  with torch.no_grad():
    for i, batch in enumerate(test_loader):
        image = batch['image']
        labels = batch['label']
        name = batch['name']
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(image)
        loss = criterion(logits.squeeze(), labels.squeeze())
        logits = torch.sigmoid(logits)
        print(f'[{i}/{len(test_loader)}] batch at loss: {loss.item()}')
        total_loss += loss.item()
        preds.append(logits.cpu().numpy())
        print(np.max(logits.squeeze().cpu().numpy()))
        save_skel(save_path + '/skel', save_path + '/pred', logits.squeeze().cpu().numpy(), name)
  
  print(total_loss/len(test_loader))
  

if __name__ == "__main__":
    test_main(args)