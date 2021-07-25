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

parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--wd", type=float, default=0.00001, help="weight decay")
parser.add_argument("--batch", type=int, default=2, help="Batch size")
parser.add_argument("--dir", type=str, default="data", help="Data directory")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")


parser.add_argument("--vals", type=int, default=1, help="The number of validation samples to test")
parser.add_argument("--val-fq", default=1, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--summary-dir", type=str, default="summary", help="Summary directory")
parser.add_argument("--log-fq", default=1, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-fq", default=50, type=int, help="How frequently to print progress to the command line in number of steps")
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
        self.summary_writer = summary_writer
        self.losses = []
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

                #print(torch.max(images), torch.max(labels))


                # print(images.size(), labels.size())#, weighting.type())

                # print(images.requires_grad)

                data_load_end_time = time.time()

                self.optimizer.zero_grad()


                logits = self.model(images)

                logit_clone = logits.detach().clone()

                max_val = torch.max(torch.sigmoid(logit_clone))
                print(max_val)


                #print(torch.max(torch.sigmoid(logits)))
               # logits = logits.view(-1,logits.size(2), logits.size(3))

                
                loss = self.criterion(logits.squeeze().double(), labels.squeeze().double())
                
                
                # if self.step == 2:
                #  get_dot = register_hooks(logits)
                
                #print("grad is ", list(self.model.parameters())[0].grad)
                #a = list(self.model.parameters())[0].clone()
                loss.backward()

               # if self.step == 2:
                    # print("here")
                   #dot = get_dot()
                 #   dot.save('tmp.dot')

                # for param in list(self.model.parameters()):
                #   print(param.requires_grad)
                
                #nn.utils.clip_grad_value_(self.model.parameters(), 0.02)
                #plot_grad_flow2(self.model.named_parameters())

                self.optimizer.step()

                self.losses.append(loss.item())
                # if self.step % 100 == 0:

                # for name, param in self.model.named_parameters():
                #     print(name, param.grad.norm())
                    # if param.grad == None:
                    #   print(name)
                

                # b = list(self.model.parameters())[0].clone()
                # print(torch.equal(a.data, b.data))

                #logits = torch.sigmoid(logits)

                # conf = gather_data(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
                # acc = (conf['tp']+conf['tn'])/(conf['tp']+conf['fp']+conf['fn']+conf['tn'])
                #print(round(acc, 5))
                #self.losses.append(loss.item())

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                # if ((self.step + 1) % args.log_fq) == 0:
                #     self.log_metrics(epoch, loss.item(), data_load_time, step_time)
                if ((self.step + 1) % args.print_fq) == 0:
                    #avg = sum(self.losses)/len(self.losses)
                    #loss = round(avg, 5)
                    self.print_metrics(epoch, stats.mean(self.losses), data_load_time, step_time)
                    self.losses.clear()

                # if self.step % 30 == 0:
                #     for tag, value in self.model.named_parameters():
                #         tag = tag.replace('.', '/')
                        # print('weights/' + tag + " ", value.data.norm().item())
                        # print('grads/' + tag + " ", value.grad.data.norm().item())
                        # self.summary_writer.add_scalar('weights/' + tag, value.data.norm().item(), self.step)
                        # self.summary_writer.add_scalar('grads/' + tag, value.grad.data.norm().item(), self.step)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            
            if((self.step + 1) % args.val_fq) == 0:
                    self.validate()
                    # self.validate() will put the model in validation mode,
                    # so we have to switch back to train mode afterwards
                    self.model.train()

            self.checkpoint(self.model, self.valloss, epoch)

            
    
    def validate(self):
        preds = []
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                print(images.dtype)
                logits = self.model(images)
                loss = self.criterion(logits.squeeze().double(), labels.squeeze(1).double())
                total_loss += loss.item()
                print("max val: ",np.max(torch.sigmoid(logits).cpu().numpy()))
                preds.append(logits.cpu().numpy())
        

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
          f"UNET_"
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

    print(torch.__version__)

    dir_train = os.readlink('scratch') + "/data/train/"


    aug_dict = dict(rotation_range=2,
                     width_shift_range=0.02,
                     height_shift_range=0.02,
                     shear_range=2,
                     zoom_range=0.02,
                     brightness_range=[0.9,1.1],
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode="nearest")

 
    flips = Flip(0.5, 0.5)
    brightness = AdjustBrightness((0.9,1.1))
    affine = Affine(translate = [(-0.02, 0.02), (-0.02, 0.02)], shear_range=(-2,2), scale=0.02, angle=(-2,2))

    transform = Transform(flips, brightness, affine)

    # transform = transforms.Compose([transforms.ToTensor()])

    # train_dataset =CoralDataset2D(sample_dir="data/train/image", 
    #                             label_dir="data/train/label",
    #                             transform=transform)

    train_dataset= CoralDatasetNew(dir_train,transform, 0)

    n_val = int(len(train_dataset)*0.05)
    n_train = len(train_dataset) - n_val
    train, val = random_split(train_dataset, [n_train, n_val])
    #validation_dataset = CoralDataset(dir_val,transform, 1)
    train_loader = DataLoader(train, batch_size=args.batch ,shuffle=True,pin_memory=True ,num_workers=args.worker_count)
    validation_loader = DataLoader(val, batch_size=1 ,shuffle=False,pin_memory=True ,num_workers=args.worker_count)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(log_dir, flush_secs=5)

    model = UNet(1, 1)
    model.apply(initialize_parameters)

    model_checkpoint = ModelCheckpoint(args)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)#, eps=1e-07, weight_decay= args.wd)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()#weight = torch.Tensor([0.1,1.0]).to(DEVICE))
    trainer = Trainer(model, model_checkpoint, criterion,optimizer, DEVICE, summary_writer, train_loader=train_loader, val_loader=validation_loader)
    trainer.train(args)
    #model = Sensor(2,1)

    # ### CHECKPOINT - load parameters, args, loss ###
    # if args.resume_checkpoint != None and args.resume_checkpoint.exists():
    #     checkpoint = torch.load(args.resume_checkpoint)
    #     print(f"Resuming model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")
    #     model.load_state_dict(checkpoint['model'])
    #     old_epochs = args.epochs
    #     args = checkpoint['args']
    #     args.epochs -= old_epochs
    
    # model_checkpoint = ModelCheckpoint(args)

    # optimizer = optim.Adam(model.parameters(), lr = args.lr)
    # criterion = nn.BCEWithLogitsLoss()
    # val_criterion = nn.BCEWithLogitsLoss()
    # trainer = Trainer(model, train_loader, validation_loader, criterion, val_criterion ,optimizer,summary_writer,model_checkpoint, DEVICE)
    # trainer.train(args)

    print("done training")

    summary_writer.close()
    
if __name__ == '__main__':
    main(parser.parse_args())