import argparse
from data import *
from models import *
#from models2 import *
from models_test import *
from model_ablated import *
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
from ctypes import CDLL
import ctypes
from image_boundaries import *

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
parser.add_argument("--log-fq", default=5, type=int, help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("--print-fq", default=10, type=int, help="How frequently to print progress to the command line in number of steps")
parser.add_argument("--tests", type=int, default=56, help="The number of tests to carry out once training is complete")

### CHECKPOINT ###
parser.add_argument("--checkpoint-path", type=Path, default=Path("./scratch/checkpoint_unet/checkpoint"))
#parser.add_argument("--checkpoint-n", type=str, default="2")
parser.add_argument("--checkpoint-fq", type=int, default=1, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path)


#parser.add_argument("--ablated", action="store_true", help="Use ablated architecture")
#parser.add_argument("--steps", type=int, default=500, help="Number of batches seen per epoch")
#parser.add_argument("--size", type=int, default=256, help="Size to reshape the images to when training")

args = parser.parse_args()


ret = os.system("cc -fPIC -shared -std=c99 -o accuracy.so accuracy.c -lm")

if ret == 0:
    print("Successfully compiled C library.")
    C = CDLL(os.path.abspath("accuracy.so"))
else:
    print("Couldn't compile C library. Exiting...")
    exit()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class HausdorfLoss(nn.Module):

    def __init__(self, reduction="mean",gamma=1, theta=1.0, sigma=1.0):
        super(HausdorfLoss, self).__init__()
        self.eps = 1e-8
        self.reduction = reduction
        self.theta = theta
        self.sigma = sigma
        self.gamma =gamma

    
    def forward(self, input, target):

        probs = torch.sigmoid(input)

        pred_clone = probs.detach().cpu().numpy()
        label_clone = target.detach().cpu().numpy()

        weight_maps = []

        for batch_ind in range(pred_clone.shape[0]):

            image_boundaries, label_boundaries = get_boundaries(pred_clone[batch_ind], label_clone[batch_ind])

            c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
            c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)
            
            C.one_value_euclidean.restype = ctypes.c_double

            weight_map = np.ones((pred_clone.shape[1], pred_clone.shape[2]))

            for i in range(0,len(label_boundaries),2):
                x = ctypes.c_int(label_boundaries[i])
                y = ctypes.c_int(label_boundaries[i+1])

                distance = C.one_value_euclidean(x,y,c_image_boundaries, len(image_boundaries))

                weight_map[label_boundaries[i]][label_boundaries[i+1]] += self.theta*math.exp(-distance/self.sigma)
            
            
            for i in range(0,len(image_boundaries),2):
                x = ctypes.c_int(image_boundaries[i])
                y = ctypes.c_int(image_boundaries[i+1])

                distance = C.one_value_euclidean(x,y,c_label_boundaries, len(c_image_boundaries))

                weight_map[image_boundaries[i]][image_boundaries[i+1]] += self.theta*math.exp(-distance/self.sigma)

            weight_maps.append(weight_map)
        
        weight_maps = torch.tensor(np.stack(weight_maps, axis=0)).to(DEVICE)

        loss_tmp = weight_maps*(-torch.pow((1. - probs), self.gamma) * target * torch.log(probs + self.eps) -torch.pow(probs, self.gamma) * (1. - target) * torch.log(1. - probs + self.eps)) #first line when target is positive class, second line when negative class

        loss_tmp = loss_tmp.squeeze(dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        
        return loss

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_gen,
        val_gen,
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
        self.val_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.step = 0
        self.losses = []
        self.summary_writer = summary_writer
        self.traindata = train_gen
        self.valdata = val_gen
        self.losses = []
        self.valloss = None
        self.train_gen = train_gen.generator()
        self.val_gen = val_gen.generator()
    
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

                # logit_clone = logits.detach().clone()

                # max_val = torch.max(torch.sigmoid(logit_clone))
                # print(max_val)


                #print(torch.max(torch.sigmoid(logits)))
               # logits = logits.view(-1,logits.size(2), logits.size(3))

                
                loss = self.criterion(logits.squeeze().double(), labels.squeeze().double())
               # accuracy = validation_accuracy(torch.sigmoid(logits).detach().cpu().numpy(), labels.detach().cpu().numpy())
                
                
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

                self.step += 1
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
                if ((self.step + 1) % args.log_fq) == 0:
                    self.log_metrics(epoch, loss.item(), data_load_time, step_time)
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

                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            
            if((self.step + 1) % args.val_fq) == 0:
                    self.validate()
                    # self.validate() will put the model in validation mode,
                    # so we have to switch back to train mode afterwards
                    self.model.train()

            self.checkpoint(self.model, self.valloss, epoch)

    
    def validate(self):
        total_loss = 0
        t_loss = 0
        accuracies = []
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits.double().squeeze(), labels.double().squeeze())
                loss_2 = nn.BCEWithLogitsLoss()(logits.squeeze(), labels.squeeze())
                total_loss += loss.item()
                t_loss += loss_2.item()
                output = torch.sigmoid(logits).cpu().numpy()

                print("max val: ",np.max(output))
                print("avg val: ",np.mean(output))
                accuracy = validation_accuracy(output, labels.cpu().numpy())
                accuracies.append(accuracy)
                print("accuracy: ",accuracy)

        

        average_loss = total_loss / len(self.val_loader)
        average_accuracy = stats.mean(accuracies)

        self.valloss = average_loss
        int_loss = t_loss / len(self.val_loader)

        print(f"average accuracy: {average_accuracy:.5f}")
        print(f"bce validation loss: {int_loss:.5f}")
        print(f"validation loss: {average_loss:.5f}")

        self.summary_writer.add_scalars(
            "loss",
            {"validation": average_loss},
            self.step
        )
        self.summary_writer.add_scalars(
            "accuracy",
            {"validation": average_accuracy},
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

    dir_train =  args.dir + "/train"
    dir_val = args.dir + "/val"


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

    transform = Transform2D(flips, brightness, affine)

    # transform = transforms.Compose([transforms.ToTensor()])

    # train_dataset =CoralDataset2D(sample_dir="data/train/image", 
    #                             label_dir="data/train/label",
    #                             transform=transform)
    
    train_dataset= CoralDataset(dir_train,transform, 0, aug_dict=aug_dict)
    validation_dataset = CoralDataset(dir_val,transform, 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch ,shuffle=True,pin_memory=True ,num_workers=args.worker_count)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch ,shuffle=False,pin_memory=True ,num_workers=args.worker_count)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(log_dir, flush_secs=5)

    model = UNetAblated(1, 1)

    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cuda'))
        else:
            # if CPU is used
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

        print(f"Testing model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")

        model.load_state_dict(checkpoint['model'])
    else:
        model.apply(initialize_parameters)

    model_checkpoint = ModelCheckpoint(args)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)#, eps=1e-07, weight_decay= args.wd)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = HausdorfLoss(sigma=10.0, theta=8.0)
    criterion = HausdorfLoss(gamma=1 ,theta=8, sigma=10)
    #criterion = nn.CrossEntropyLoss()#weight = torch.Tensor([0.1,1.0]).to(DEVICE))
    trainer = Trainer(model, train_dataset, validation_dataset, model_checkpoint, criterion,optimizer, DEVICE, summary_writer, train_loader=train_loader, val_loader=validation_loader)
    trainer.train(args)
    #model = Sensor(2,1)


    print("done training")

    summary_writer.close()


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
    
if __name__ == '__main__':
    main(parser.parse_args())