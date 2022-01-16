import argparse
from data_final import *
from model_final import *
from model_checkpoint_final import *
from transformClass_final import *
from loss_functions_final import *
import os
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np

import statistics as stats

import time

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.00001, help="weight decay")
parser.add_argument("--batch", type=int, default=2, help="Batch size")
parser.add_argument("--dir", type=str, default="../../data/", help="Data directory")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")


parser.add_argument("--vals", type=int, default=1, help="The number of validation samples to test")
parser.add_argument("--val-fq",type=int, default=1, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--summary-dir", type=str, default="summary", help="Summary directory")
parser.add_argument("--log-fq", default=1, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-fq", default=1, type=int, help="How frequently to print progress to the command line in number of steps")

### CHECKPOINT ###
parser.add_argument("--checkpoint-path", type=Path, default=Path("../../checkpoint/checkpoint_folder"))
parser.add_argument("--checkpoint-fq", type=int, default=1, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path)

# Transform args
parser.add_argument("--flip-range", type=float, default=0.5, help="flipping range +/- value")
parser.add_argument("--brightness-range", type=float, default=0.1, help="range of shifts of brightness +/- value")
parser.add_argument("--translation-range", type=float, default=0.2, help="translation +/- value")
parser.add_argument("--shear-range", type=float, default=2, help="shearing range")
parser.add_argument("--angle", type=int, default=2, help="rotation angle +/- value")
parser.add_argument("--scale", type=float, default=0.02, help="scaling value")

#Focal loss args
parser.add_argument("--alpha", type=int, default=0.5, help="alpha value used for focal loss")
parser.add_argument("--gamma", type=float, default=1, help="gamma value used for focal loss")

#UNET agrs
parser.add_argument("--n_channels", type=int, default=1, help="number of input channels")
parser.add_argument("--n_classes", type=int, default=1, help="number of output channels")

args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class Trainer:
    def __init__(
        self, 
        model, 
        checkpoint,
        start_epoch,
        criterion, 
        optimizer,
        device,
        train_loader,
        val_loader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint = checkpoint
        self.start_epoch = start_epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.losses = []
        self.valloss = None
    
    def train(self, args):
        

        self.model.train()
        
        for epoch in range(self.start_epoch, args.epochs):
            data_load_start_time = time.time()

            print("start epoch")

            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                data_load_end_time = time.time()

                self.optimizer.zero_grad()


                logits = self.model(images)

                
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                
                loss.backward()

                self.optimizer.step()

                self.losses.append(loss.item())

                self.step += 1

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step) % args.print_fq) == 0:
                    self.print_metrics(epoch, stats.mean(self.losses), data_load_time, step_time)
                    self.losses.clear()

                data_load_start_time = time.time()
            
                if (self.step % args.val_fq) == 0:
                    self.validate()
                    self.checkpoint(self.model, self.valloss, epoch)
                    self.model.train()

            
    
    def validate(self):
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                total_loss += loss.item()
        

        average_loss = total_loss / len(self.val_loader)

        self.valloss = average_loss

        print(f"validation loss: {average_loss:.5f}")

    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], " 
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
    
    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss)},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def main(args):

    dir_train = args.dir + "/train/"
    dir_val = args.dir + "/val/"
    flip_range = args.flip_range
    brightness_range = args.brightness_range
    translation_range = args.translation_range
    shear_range = args.shear_range
    scale = args.scale
    angle = args.angle
    start_epoch = 0

 
    flips = Flip(flip_range, flip_range)
    brightness = AdjustBrightness((1-brightness_range,1+brightness_range))
    affine = Affine(translate = [(-translation_range, translation_range), (-translation_range, translation_range)], shear_range=(shear_range,shear_range), scale=scale, angle=(-angle,angle))

    transform = Transform(flips, brightness, affine)
    
    train_dataset= CoralDataset(dir_train,transform, 0)
    val_dataset = CoralDataset(dir_val,transform,1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch ,shuffle=True,pin_memory=True ,num_workers=args.worker_count)
    validation_loader = DataLoader(val_dataset, batch_size=1 ,shuffle=False,pin_memory=True ,num_workers=args.worker_count)

    model = UNetAblated(args.n_channels, args.n_classes)
    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

        print(f"Testing model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")

        model.load_state_dict(checkpoint['model'])
        start_epoch = int(args.resume_checkpoint.split("-")[1])+1
    else:
        model.apply(initialize_parameters)
        

    print(f"starting at epoch: {start_epoch}")
    model_checkpoint = ModelCheckpoint(args)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)
    trainer = Trainer(model, model_checkpoint, start_epoch, criterion, optimizer, DEVICE, train_loader=train_loader, val_loader=validation_loader)
    trainer.train(args)


    print("done training")

    
if __name__ == '__main__':
    main(parser.parse_args())

