import argparse
from data import *
from models import *
#from models2 import *
from models_test import *
from tools import *
from transformClass import *
from sensor import *
from sensorAblated import *
from loss_functions import *
import os
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np

import statistics as stats

import time

torch.backends.cudnn.benchmark = True

#All possible arguments.
parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.00001, help="weight decay")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--dir", type=str, default="data", help="Data directory")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")


parser.add_argument("--vals", type=int, default=1, help="The number of validation samples to test")
parser.add_argument("--val-fq", default=161, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--summary-dir", type=str, default="summary", help="Summary directory")
parser.add_argument("--log-fq", default=1, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-fq", default=10, type=int, help="How frequently to print progress to the command line in number of steps")
parser.add_argument("--tests", type=int, default=56, help="The number of tests to carry out once training is complete")

### CHECKPOINT ###
parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoint/checkpoint_lstm"))
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


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        checkpoint,
        criterion: nn.Module, 
        optimizer,
        device: torch.device,
        summary_writer,
        train_loader,
        val_loader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint = checkpoint
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.losses = []
        self.summary_writer = summary_writer
        self.valloss = None
    
    def train(self, args):
        

        self.model.train()

        # l = [module for module in self.model.modules() if type(module) != nn.Sequential]

        # print(l)

        
        for epoch in range(args.epochs):
            data_load_start_time = time.time()

            print("start epoch")

            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                data_load_end_time = time.time()

                self.optimizer.zero_grad()


                logits = self.model(images)
            
                # logit_clone = logits.detach().clone()

                # max_val = torch.max(torch.sigmoid(logit_clone))
                # print(torch.mean(torch.sigmoid(logit_clone)))
                # print(max_val)

                # for name, param in self.model.named_parameters():
                #     print(name)

                #print(f'out size is {logits.size()}')

                
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                
                loss.backward()
                self.optimizer.step()

                self.losses.append(loss.item())

                self.step += 1

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step) % args.log_fq) == 0:
                    self.log_metrics(epoch, loss.item(), data_load_time, step_time)
                if ((self.step) % args.print_fq) == 0:
                    self.print_metrics(epoch, stats.mean(self.losses), data_load_time, step_time)
                    self.losses.clear()

                data_load_start_time = time.time()

             #   self.summary_writer.add_scalar("epoch", epoch, self.step)
            
                if((self.step) % args.val_fq) == 0:
                    self.validate()
                        # self.validate() will put the model in validation mode,
                        # so we have to switch back to train mode afterwards
                    self.model.train()

                    self.checkpoint(self.model, self.valloss, epoch)

            
    
    def validate(self):
        total_loss = 0
        t_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits.squeeze(), labels.squeeze())
                total_loss += loss.item()
                print("max val: ",np.max(torch.sigmoid(logits).cpu().numpy()))
                print("avg val: ",np.mean(torch.sigmoid(logits).cpu().numpy()))
        

        average_loss = total_loss / len(self.val_loader)

        self.valloss = average_loss

        print(f"validation loss: {average_loss:.5f}")

        self.summary_writer.add_scalars(
            "loss",
            {"test": average_loss},
            self.step
        )

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

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
          f"SENSOR_"
          f"bs={args.batch}_" +
          f"lr={args.lr}_" +
          f"run_"
      )
    
    print(tb_log_dir_prefix)
    i = 0
    while i < 1000:
        tb_log_dir = Path(args.summary_dir + "/" + (tb_log_dir_prefix + str(i)))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        #print(m)
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
                

def main(args):

    dir_train = "D:/2D-remake/3ddata/chunk1/train"
    #dir_train = os.readlink('scratch') + "/train/"

 
    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine, mode='3D')
    
    train_dataset= CoralDataset3D(dir_train,transform, 0, k=3)
    n_val = 50
    n_train = len(train_dataset) - n_val
    train, val = random_split(train_dataset, [n_train, n_val])
    #validation_dataset = CoralDataset(dir_val,transform, 1)
    train_loader = DataLoader(train, batch_size=args.batch ,shuffle=True,pin_memory=True ,num_workers=args.worker_count)
    validation_loader = DataLoader(val, batch_size=1 ,shuffle=False,pin_memory=True ,num_workers=args.worker_count)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(log_dir, flush_secs=5)

    #model = UNet(1, 1)
    model = SensorAblated(1,1)
    model.apply(initialize_parameters)

    model_checkpoint = ModelCheckpoint(args)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)#, eps=1e-07, weight_decay= args.wd)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    #criterion = nn.BCEWithLogitsLoss()# 
    criterion = FocalLoss(gamma=1, alpha=0.1)
    #criterion = nn.CrossEntropyLoss()#weight = torch.Tensor([0.1,1.0]).to(DEVICE))
    trainer = Trainer(model, model_checkpoint, criterion,optimizer, DEVICE, summary_writer, train_loader=train_loader, val_loader=validation_loader)
    trainer.train(args)
    #model = Sensor(2,1)


    print("done training")

    summary_writer.close()
    
if __name__ == '__main__':
    main(parser.parse_args())