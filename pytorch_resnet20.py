from __future__ import print_function # What's for

# wandb
import os
import wandb
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms

from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import horovod.torch as hvd

from grace_dl.torch.helper import grace_from_params

from model import ResNet


# Training configuritions
parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='size of each batch of cifar-10 training images (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
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
print('torch.cuda.is_available() is {}'.format(torch.cuda.is_available()))
args.log_interval = 100
# use gpu for training
if not torch.cuda.is_available():
    print('Error: CUDA library unavailable on system')
    os._exit()
    # return
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Horovod: initialize library
hvd.init()
torch.manual_seed(args.seed)    # why we set the random seed? Just as an example?

# Test: print the rank of each GPU (not the local rank, what is the difference? The local rank will print something like an address)
print('This is the rank {} device.'.format(hvd.rank()))

# Set fields
# Test: print the rank of each GPU (not the local rank, what is the difference? local rank will give something like an address)
print('The compression ratio is {}.'.format(args.compression_ratio))
# compressor = 'topk'
compressor = 'baseline'
run_name = 'ResNet-worker-' + str(hvd.rank()) + '-' + compressor
# run_name = 'ResNet-worker-' + str(hvd.rank()) + '-' + compressor + '-' + str(args.compression_ratio)

# Initialize wandb
wandb.init(project="project-csens", name=run_name)
# You must call wandb.init() before wandb.config.*
# Save model inputs and hyperparameters
config = wandb.config
config.batch_size = args.batch_size
config.test_batch_size = args.test_batch_size
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
# transform_augment and transform_normalize are list of Transform objects
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

NUM_TRAIN = 45000
NUM_VAL = 5000
train_dataset_origin = \
     datasets.CIFAR10('./dataset-%d' % hvd.rank(), train=True, download=True,
                transform=transforms.Compose([transform_normalize]))
val_list = list(range(NUM_TRAIN, NUM_TRAIN+NUM_VAL))
# Extract a subset for validation from the original train set.
val_dataset = torch.utils.data.Subset(train_dataset_origin, val_list)  
# Horovod: use DistributedSampler to partition the val data.    
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.test_batch_size, sampler=val_sampler, **kwargs)

test_dataset = \
     datasets.CIFAR10('./dataset-%d' % hvd.rank(), train=False, download=True,
                transform=transforms.Compose([transform_normalize]))
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())     # need shuffle or not? default is True
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

    
# load model
model = ResNet(args.n, res_option=args.res_option, use_dropout=args.use_dropout)

def get_param_count(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    return sum(param_counts)

param_count = get_param_count(model)
print('Parameter count: %d' % param_count)

# Check how many model will be instantiated 
print('An instance of ResNet() has been instantiated.')
print('This is the rank {} device.'.format(hvd.rank()))

# Logs metrics with wandb
wandb.watch(model)

if args.cuda:
    # Move model to GPU.
    model.cuda()
      
# GRACE: compression algorithm.

# params = {'compressor': 'topk', 'compress_ratio': args.compression_ratio, 'memory':'residual', 'communicator':'allgather', 'world_size':hvd.size()}
params = {}
# grc = Allgather(TopKCompressor(config.compression_ratio), ResidualMemory(), hvd.size())
grc = grace_from_params(params)


    # global gpu_dtype
    # gpu_dtype = torch.cuda.FloatTensor
    # model = model.type(gpu_dtype)
    
    # setup loss function
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def check_accuracy(mode):
    
    model.eval()
    
    print('mode is {}'.format(mode))

    loader_t = val_loader if mode == 'val' else test_loader
    sampler_t = val_sampler if mode == 'val' else test_sampler
    
    test_loss = 0.
    test_accuracy = 0.
    for data, target in loader_t:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        print('size of output is {}'.format(list(output.size())))
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
    
    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(sampler_t)
    test_accuracy /= len(sampler_t)
    
    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\n' + mode + ' set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))
        # Log matrics with wandb
        if mode == 'val':
            wandb.log({"Val Accuracy": 100. * test_accuracy, "Val Loss": test_loss})
        else:
            wandb.log({"Final Test Accuracy": 100. * test_accuracy, "Test Loss": test_loss})



def train(epoch):
    
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    
    # test:0
    iters = len(train_loader) # total number of batches in this epoch
    print('-' * 60)
    print('current epoch is {}'.format(epoch))
    print('length of the train_loader(number of iteration per epoch) is {} '.format(iters))  # 235 idexing from (0, 234)
    print('-' * 60)
    
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch) # might have a problem here!!!! see what is this set_epoch means
    for batch_idx, (data, target) in enumerate(train_loader):

        print('batch_idx is {}'.format(batch_idx))    

        if args.cuda:
            data, target = data.cuda(), target.cuda()   # change the output into the cude form?
        
        # set the gradients to zero
        optimizer.zero_grad()

        # predict the output (forward propagation)   
        output = model(data) 

        # F is for torch.nn.functional
        loss = F.cross_entropy(output, target) 

        # backward propagation: compute the grads of the loss w.r.t. the model's parameters
        loss.backward() 

        # update the NN weights, the optimizer has been wrapped by horovod, so I believe allreduce is performed here 
        optimizer.step()

        if batch_idx % args.log_interval == 0:

            commit_flag = True
            # commit_flag = False if batch_idx == 230 else True # (234/10) * 10  

            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"Train Loss": loss.item()}, commit=commit_flag)


# train model
SCHEDULE_EPOCHS = [50, 5, 5] # divide lr by 10 after each number of epochs
acc_epochs = 0
#     SCHEDULE_EPOCHS = [100, 50, 50] # divide lr by 10 after each number of epochs
learning_rate = args.lr # initial learning rate, by default is 0.1

# Broadcast parameters
hvd.broadcast_parameters(model.state_dict(), root_rank=0)


for num_epochs in SCHEDULE_EPOCHS:

    print('-' * 60)
    print('Training for %d epochs with learning rate %f' % (num_epochs, learning_rate))
    print('-' * 60)

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(),
                            momentum=0.9, weight_decay=args.weight_decay)
    # Horovod: broadcast parameters & optimizer state.
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, grc, named_parameters=model.named_parameters())
    
    for epoch in range(acc_epochs, acc_epochs + num_epochs):
        
        # compute the validation accuracy at the beginning of each epoch
        check_accuracy('val')
        print('Starting epoch %d / %d' % (epoch+1, num_epochs))
        # train the model
        train(epoch)
        # try to put 

    # after num_epochs epoches decrease the lr
    learning_rate *= 0.1
    acc_epochs += num_epochs


print('Final test accuracy:')
check_accuracy('test')

# WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
print('wand.run.dir is {}'.format(wandb.run.dir))

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
wandb.save('model.pt')
#end