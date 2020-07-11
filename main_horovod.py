from __future__ import print_function # What's for

# wandb
import os
import wandb
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler


# import torchvision.datasets as dset
# import torchvision.transforms as T

import horovod.torch as hvd

from grace_dl.torch.helper import grace_from_params

from model import ResNet



# Training configuritions
parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='size of each batch of cifar-10 training images (default: 64)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--weight-decay', type=float, default=0.0001,  # may need to change
                    help='parameter to decay weights')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,        # what is fp16 compression during allreduce
                    help='use fp16 compression during allreduce')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',                   # the default random seed is 42
                    help='random seed (default: 42)')
# ResNet parameters
parser.add_argument('-n', default=5, type=int,
                    help='value of n to use for resnet configuration (see https://arxiv.org/pdf/1512.03385.pdf for details)')
parser.add_argument('--use-dropout', default=False, const=True, nargs='?',
                    help='whether to use dropout in network')
parser.add_argument('--res-option', default='A', type=str,
                    help='which projection method to use for changing number of channels in residual connections')
# Data directory
parser.add_argument('--data-dir', default='./dataset', type=str,
                    help='path to dataset')
# Compression ratio
parser.add_argument('--compression-ratio', type=float, default=0.3, metavar='CR',   
                    help='the compression ratio the topk algorithm uses')                 
args = parser.parse_args()

# test
print('args.no_cuda is {}'.format(args.no_cuda))
print('torch.cuda.is_available() is {}'.format(torch.cuda.is_available()))
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library
hvd.init()
torch.manual_seed(args.seed)    # why we set the random seed? Just as an example?

# Test: print the rank of each GPU (not the local rank, what is the difference? The local rank will print something like an address)
print('This is the rank {} device.'.format(hvd.rank()))

# Set fields
# Test: print the rank of each GPU (not the local rank, what is the difference? local rank will give something like an address)
print('The compression ratio is {}.'.format(args.compression_ratio))
compressor = 'topk'
run_name = 'worker-' + str(hvd.rank()) + '-' + compressor + '-' + str(args.compression_ratio)

# Initialize wandb
wandb.init(project="project-csens", name=run_name)
# You must call wandb.init() before wandb.config.*
# Save model inputs and hyperparameters
config = wandb.config
config.batch_size = args.batch_size
# config.epochs = args.epochs
config.lr = args.lr
# config.momentum = args.momentum
config.no_cuda = args.no_cuda
config.seed = args.seed
config.log_interval = args.log_interval

config.compressor = compressor
config.compression_ratio = args.compression_ratio

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)      # seed is default set to 42

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(1)

# define transforms for normalization and data augmentation
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4)])
transform_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = \
     datasets.CIFAR10('./dataset-%d' % hvd.rank(), train=True, download=True,
                transform=transforms.Compose([
                    transform_augment,
                    transform_normalize
                ]))
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)  # batch size is 128 (default)


val_dataset = \
     datasets.CIFAR10('./dataset-%d' % hvd.rank(), train=True,
                transform=transforms.Compose([
                    transform_augment, 
                    transform_normalize
                ]))
loader_val = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# Horovod: use DistributedSampler to partition the validation data.         


test_dataset = \
     datasets.CIFAR10('./dataset-%d' % hvd.rank(), train=False,
                transform=transforms.Compose([
                    transform_augment, 
                    transform_normalize
                ]))
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)

# get CIFAR-10 data
NUM_TRAIN = 45000
NUM_VAL = 5000


# def main(args):


   
    
    # load model
    model = ResNet(args.n, res_option=args.res_option, use_dropout=args.use_dropout)
    
    # will call the get_param_count function below
    param_count = get_param_count(model)
    print('Parameter count: %d' % param_count)
    
    # use gpu for training
    if not torch.cuda.is_available():
        print('Error: CUDA library unavailable on system')
        return
    global gpu_dtype
    gpu_dtype = torch.cuda.FloatTensor
    model = model.type(gpu_dtype)
    
    # setup loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # train model
    SCHEDULE_EPOCHS = [50, 5, 5] # divide lr by 10 after each number of epochs
#     SCHEDULE_EPOCHS = [100, 50, 50] # divide lr by 10 after each number of epochs
    learning_rate = 0.1
    for num_epochs in SCHEDULE_EPOCHS:
        print('Training for %d epochs with learning rate %f' % (num_epochs, learning_rate))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(num_epochs):
            
            # compute the validation accuracy at the beginning of each epoch
            check_accuracy(model, loader_val)
            print('Starting epoch %d / %d' % (epoch+1, num_epochs))

            # train the model
            train(loader_train, model, criterion, optimizer)
        # after num_epochs epoches decrease the lr
        learning_rate *= 0.1
    
    print('Final test accuracy:')
    check_accuracy(model, loader_test)

def check_accuracy(model, loader):

    num_correct = 0
    num_samples = 0
    model.eval()

    for X, y in loader:
        X_var = Variable(X.type(gpu_dtype), volatile=True)

        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)
        # test
        print('socres size is {}, preds size is {}, y size is {}'.format(scores.size, preds.size, y.size))

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(loader_train, model, criterion, optimizer):
    model.train()
    # test
    iters = len(loader_train)
    print('The number of iterations in each epoch is {}'.format(iters))
    
    for t, (X, y) in enumerate(loader_train):


        X_var = Variable(X.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype)).long()

        scores = model(X_var)
        # criterion is the way we calculate the loss
        loss = criterion(scores, y_var) 
        if (t+1) % args.log_interval == 0:
            # print('t = %d, loss = %.4f' % (t+1, loss.data[0]))
            # is loss.data[0] a 0-dim tensor?
            print('t = %d, loss = %.4f' % (t+1, loss.item()))
        
        # set the gradients to zero
        optimizer.zero_grad()
        
        # backward propagation: compute the grads
        loss.backward()
        
        # update the NN weights, the optimizer has been wrapped by horovod, so I believe allreduce is performed here 
        optimizer.step()

def get_param_count(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    return sum(param_counts)

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples # num train 45000
        self.start = start             # start 0
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples

# if __name__ == '__main__':
    # args = parser.parse_args()
    # main(args)
