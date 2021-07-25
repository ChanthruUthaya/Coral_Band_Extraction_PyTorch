import argparse
from data import *
from models import *
#from models2 import *
from models_test import *
from tools import *
from transformClass import *
from sensor3d import *
from loss_functions import *
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
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

import statistics as stats

import time

torch.backends.cudnn.benchmark = True

#All possible arguments.
parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--batch", type=int, default=2, help="Batch size")
parser.add_argument("--dir", type=str, default="data", help="Data directory")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")


parser.add_argument("--vals", type=int, default=1, help="The number of validation samples to test")
parser.add_argument("--val-fq", default=1, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--summary-dir", type=str, default="summary", help="Summary directory")
parser.add_argument("--log-fq", default=1, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-fq", default=1, type=int, help="How frequently to print progress to the command line in number of steps")
parser.add_argument("--tests", type=int, default=56, help="The number of tests to carry out once training is complete")

### CHECKPOINT ###
parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoint/checkpoint"))
#parser.add_argument("--checkpoint-n", type=str, default="2")
parser.add_argument("--checkpoint-fq", type=int, default=1, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path)


#parser.add_argument("--ablated", action="store_true", help="Use ablated architecture")
#parser.add_argument("--steps", type=int, default=500, help="Number of batches seen per epoch")
#parser.add_argument("--size", type=int, default=256, help="Size to reshape the images to when training")

args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class tempDataset(Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.img_dir = dir+"/image"
        self.label_dir = dir + "/label"
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)][:1]

    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            image = image / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return Image.fromarray(image), Image.fromarray(label)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, i):

        label_file = glob(self.label_dir + '/' + self.ids[i] + "_label" + '.*')[0]
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0]

        img = Image.open(img_file.replace('\\', '/'))
        label = Image.open(label_file.replace('\\', '/'))

        img, label = tempDataset.adjust_data(np.array(img), np.array(label))

        return transforms.ToTensor()(img), transforms.ToTensor()(label)



class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer,
        device: torch.device,
        train_loader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.losses = []
    
    def train(self, args):
        

        self.model.train()

        for epoch in range(args.epochs):

            print("start epoch")

            for image, labels in self.train_loader:
                images = image.to(self.device)
                labels = labels.to(self.device)


                self.optimizer.zero_grad()


                logits = self.model(images)


                loss = self.criterion(logits.squeeze(), labels.squeeze())

                loss.backward()


                self.optimizer.step()

                self.losses.append(loss.item())

                if ((self.step + 1) % args.print_fq) == 0:

                    self.print_metrics(epoch, stats.mean(self.losses))
                    self.losses.clear()
                
                self.step += 1
            

    def print_metrics(self, epoch, loss):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], " 
                f"batch loss: {loss:.5f}, "
        )


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
                

def main(args):


    dir_train =  args.dir + "/train"

    train_dataset= tempDataset(dir_train)
    train_loader = DataLoader(train_dataset, batch_size=1 ,shuffle=True,pin_memory=True ,num_workers=1)

    model = UNet(1, 1)
    model.apply(initialize_parameters)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, eps=1e-07)
    criterion = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, criterion,optimizer, DEVICE, train_loader=train_loader)
    trainer.train(args)
    print("done training")

    
if __name__ == '__main__':
    main(parser.parse_args())