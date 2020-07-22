import torch
import torch.nn as nn
import torch as th
import numpy as np
import sys
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from time import time
#from dense_ed import R_Decoder
#from Combined_VAE import VAE
from torch.autograd import Variable
#import argparse
#args.train_dir = args.run_dir + "/training"
#args.pred_dir = args.train_dir + "/predictions"
#mkdirs([args.train_dir, args.pred_dir])
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
#from Pre_load_data import pre_load_data
#from train_load_data import train_load_data
#from test_load_data import test_load_data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import matplotlib.ticker as ticker
import matplotlib
from pylab import *
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
torch.set_default_tensor_type('torch.FloatTensor')
import time
import os
import sys

tx1=time.time()
class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.

    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(p=drop_rate))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.

        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                self.add_module('convT2', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))


def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, 
                       out_channels, kernel_size=kernel_size, stride=stride, 
                       padding=padding, output_padding=output_padding, bias=bias))
    return last_up


def activation(name, *args):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 init_features=48, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None):
        """Dense Convolutional Encoder-Decoder Networks.

        In the network presented in the paper, the last decoding layer 
        (transition up) directly outputs the predicted fields. 

        The network parameters should be modified for different image size,
        mostly the first conv and the last convT layers. (`output_padding` in
        ConvT can be modified as well)

        Args:
            in_channels (int): number of input channels (also include time if
                time enters in the input)
            out_channels (int): number of output channels
            blocks (list-like): A list (of odd size) of integers
            growth_rate (int): K
            init_features (int): the number of feature maps after the first
                conv layer
            bn_size: bottleneck size for number of feature maps
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
        """
        super(Encoder, self).__init__()
        """         if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:] 
        """

        self.features = nn.Sequential()

        # First convolution, half image size ================
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        self.features.add_module('In_conv', nn.Conv2d(in_channels, init_features, 
                              kernel_size=7, stride=2, padding=2, bias=False))
        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = init_features
        for i, num_layers in enumerate(blocks):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True, 
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2

    def forward(self, x):
        return self.features(x)

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x


    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None):
        """Dense Convolutional Encoder-Decoder Networks.

        In the network presented in the paper, the last decoding layer 
        (transition up) directly outputs the predicted fields. 

        The network parameters should be modified for different image size,
        mostly the first conv and the last convT layers. (`output_padding` in
        ConvT can be modified as well)

        Args:
            in_channels (int): number of input channels (also include time if
                time enters in the input)
            out_channels (int): number of output channels
            blocks (list-like): A list (of odd size) of integers
            growth_rate (int): K
            init_features (int): the number of feature maps after the first
                conv layer
            bn_size: bottleneck size for number of feature maps
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
        """
        super(Decoder, self).__init__()
        # if len(blocks) > 1 and len(blocks) % 2 == 0:
        #     raise ValueError('length of blocks must be an odd number, but got {}'
        #                     .format(len(blocks)))
        # enc_block_layers = blocks[: len(blocks) // 2]
        # dec_block_layers = blocks[len(blocks) // 2:]

        self.features = nn.Sequential()

        # First convolution, half image size ================
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        # self.features.add_module('In_conv', nn.Conv2d(in_channels, init_features, 
        #                       kernel_size=3, stride=1, padding=1, bias=False))
        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        # num_features = init_features
        # Decoding / transition up ==============
        # dense block --> decoding --> dense block --> decoding --> dense block
        num_features = in_channels

        for i, num_layers in enumerate(blocks):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(blocks) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        # The last decoding layer =======
        last_trans_up = last_decoding(num_features, out_channels, 
                            kernel_size=4, stride=2, padding=1, 
                            output_padding=1, bias=False, drop_rate=drop_rate)
        self.features.add_module('LastTransUp', last_trans_up)
        if out_activation is not None:
            self.features.add_module(out_activation, activation(out_activation))
        
        print('# params {}, # conv layers {}'.format(
            *self._num_parameters_convlayers()))

    def forward(self, x):
        return self.features(x)

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x
    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
# if __name__ == '__main__':

#     dense_ed = Decoder(48, 3, blocks=(6,3), growth_rate=16, 
#                       drop_rate=0, bn_size=8, 
#                       bottleneck=False, out_activation='Tanh')
#     print(dense_ed)
#     x = torch.Tensor(16, 48, 16, 16)
#     dense_ed.forward_test(x)
#     print(dense_ed._num_parameters_convlayers())

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=3072, z_dim=14400):
        super(VAE, self).__init__()
        self.encoder = Encoder(1, 1, blocks=(3,), growth_rate=16, 
                       drop_rate=0, bn_size=8, 
                       bottleneck=False, out_activation='None')
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.decoder = Decoder(48, 3, blocks=(6,3), growth_rate=16, 
                       drop_rate=0, bn_size=8, 
                       bottleneck=False, out_activation='None')
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        #print('std type',std.dtype)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        #print('esp type',esp.dtype)
        #print('mu type',mu.dtype)
        #z1=std * esp
        #print('prod type',z1.dtype)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        #print ('here',h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        #h = h.view(-1,18432)
        z, mu, logvar = self.bottleneck(h.view(-1,3072))
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(16,48,8,8)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

#load pre-training data

# pre_train_loader = pre_load_data(10)

# # load data
# train_data_dir = args.data_dir + '/kle{}_lhs{}.hdf5'.format(args.kle, args.ntrain)
# test_data_dir = args.data_dir + '/kle{}_mc{}.hdf5'.format(args.kle, args.ntest)
# train_loader, train_stats = load_data(train_data_dir, args.batch_size)
# test_loader, test_stats = load_data(test_data_dir, args.test_batch_size)
print('Loaded data!')
print('Start training........................................................')
#tic = time()
#########################pre-train###########
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = torch.load('VAE_var_test3_14400.pt').to(device)
#print (vae)
#print ('here>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',vae)
#for name, param in vae.named_parameters():
   #print(name)

#changed here ------------->
batch_size = 16

#train_loader, test_loader, stats = load_data(batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24349, 0.26159))])
#(0.5, 0.5, 0.5), (0.5, 0.5, 0.5) #((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./cifardata', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./cifardata', train=False, transform=trans)

#tstart = time.time()
# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

#------>
class Regression(nn.Module):
    def __init__(self, D_in, D_out, p=0.0):
        """
        Small 2 layer dropout Neural Net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
            p (float) = Dropout probability
        """
        super(Regression, self).__init__()
        self.p = p
        self.linear1 = th.nn.Linear(D_in, 10000)
        self.f1 = th.nn.ReLU()
        self.linear2 = th.nn.Linear(10000, 6000)
        self.f2 = th.nn.ReLU()
        self.linear2a = th.nn.Linear(6000, 4000)
        self.f2a = th.nn.ReLU()
        self.linear2b = th.nn.Linear(4000, 2000)
        self.f2b = th.nn.ReLU()
        self.linear2c = th.nn.Linear(2000, 500)
        self.f2c = th.nn.ReLU()
        self.linear3 = th.nn.Linear(500, 100)
        self.f3 = th.nn.ReLU()
        self.linear3a = th.nn.Linear(100, D_out)

    def forward(self, x):
        
        lin1 = self.f1(self.linear1(x))
        lin1 = F.dropout(lin1, p=self.p, training=self.training)
        lin2 = self.f2(self.linear2(lin1))
        lin2 = F.dropout(lin2, p=self.p, training=self.training)
        lin2a = self.f2a(self.linear2a(lin2))
        lin2a = F.dropout(lin2a, p=self.p, training=self.training)
        lin2b = self.f2b(self.linear2b(lin2a))
        lin2b = F.dropout(lin2b, p=self.p, training=self.training)
        lin2c = self.f2c(self.linear2c(lin2b))
        lin2c = F.dropout(lin2c, p=self.p, training=self.training)
        lin3 = self.f3(self.linear3(lin2c))
        lin3a = F.dropout(lin3, p=self.p, training=self.training)
        out = self.linear3a(lin3a)
        
        return out

plot_freq = 100
n_out_pixels_train = 3244800
n_out_pixels_test = 6337500
model = Regression(D_in = 14400, D_out = 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5,
                       weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)
loss_c=nn.CrossEntropyLoss()
# Train the model and init variables for results
total_step = len(train_loader)
loss_list = []
acc_list = []
acc_test_list = []
loss_test_list = []
train_acc_list = []
test_acc_list = []

epochs=200 #set the epochs here

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    #mse = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        #data=Variable(input.view(-1,4225))
        target_out = target.view(16)
        #data=Variable(input.view(-1,4225))
        with torch.no_grad():
            if device.type == 'cuda':
                input=input.cuda()
                target_out=target_out.cuda()
            out, mu, var = vae.encode(input)
        output = model(out)
        loss_test = loss_c(output, target_out)
        loss_test_list.append(loss_test.item())  # append loss to list
        _, predicted = torch.max(output.data, 1)
        total += target_out.size(0)
        correct += (predicted == target_out).sum().item()
        acc_test_list.append(correct / total)
    test_acc_list.append((correct / total)*100) #saving at the end of epoch
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

def train(epoch):
    correct_train = 0
    total_train = 0
    model.train()
    #mse = 0.
    for i, (input, target) in enumerate(train_loader):
        #data=Variable(input.view(-1,4225))
        target = target.view(16)
        model.zero_grad()
        #data=Variable(input.view(-1,4225))
        with torch.no_grad():
            if device.type == 'cuda':
                input=input.cuda()
                target=target.cuda()
            out, mu, var = vae.encode(input)
        output = model(out)
        #print (target.size())
        
        loss = loss_c(output, target)
        loss_list.append(loss.item())  # append loss to list

        # Backprop and perform Adam optimisation
        loss.backward()  # backprop

        optimizer.step()  # update the weights
        #for param in model.parameters():
          #print('gradients are ',param.grad)

        # Track the accuracy
        total = target.size(0)
        _, predicted = torch.max(output.data, 1)
        #print('pred',predicted)
        correct = (predicted == target).sum().item()
        acc_list.append(correct / total)  # accuracy of o/p in list
        
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
        
        ####### print the loss and accuracy for every 100 batches ############
        if (i + 1) % batch_size == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch , epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
            # print('filters of conv layer are ',model.layer1[0].filters.data.numpy())
            # print('weights of fc layer are ',model.fc1.weight.data.numpy())
            # print('bias of fc layer are ',model.fc1.bias.data.numpy())
    train_acc_list.append((correct_train / total_train)*100) #saving at the end of epoch
    scheduler.step(loss)
    #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#epochs = 1   #24/3: changed to 1 epoch  
for epoch in range(1, epochs + 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_1 =train(epoch)
    with torch.no_grad():
        test_1 = test(epoch)
torch.save(model, 'CIFAR_reg_model.pt')
        
print('==================================================================\n Save to file\n=================================================================\n')
np.savetxt("CIFAR_Lossfunction_gpu_3x3.txt", np.array(loss_list))
np.savetxt("CIFAR_test_Lossfunction_gpu_3x3.txt", np.array(loss_test_list))
np.savetxt('CIFAR_Accuracyfunction_gpu_3x3.txt', np.array(acc_list) * 100)
np.savetxt('CIFAR_Accuracy_test_function_gpu_3x3.txt', np.array(acc_test_list) * 100)
np.savetxt('CIFAR_Accuracy_test_epoch_function_gpu_3x3.txt', np.array(test_acc_list) )
np.savetxt('CIFAR_Accuracy_train_epoch_function_gpu_3x3.txt', np.array(train_acc_list) )

#x = np.arange(0, epochs * len(train_loader), 1)  # -> x label
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#ax1.plot(x, np.array(loss_list), 'b')
#ax2.plot(x, np.array(acc_list) * 100, 'r')
#ax1.set_xlabel('#batches')
#ax1.set_ylabel('Loss ', color='b')
#ax2.set_ylabel('Accuracy (%)', color='r')
#plt.title('Reg Results')
#plt.savefig('CIFAR_figure0.png')
#
#plt.figure(2)
#x1 = np.arange(0, epochs * len(test_loader), 1)
#plt.plot(x1, np.array(acc_test_list) * 100, 'g')
#plt.xlabel('#batches')
#plt.ylabel('Test Accuracy ', color='g')
#plt.title('Test Results !!!!')
#plt.legend(loc='best', frameon=False)
#plt.savefig('CIFAR_figure1.png')

plt.figure(3)
l1 = np.array(loss_list)
l2 = np.array(loss_test_list)
c1 = np.ceil(50000/batch_size).astype(int)
c2 = np.ceil(10000/batch_size).astype(int)
x11 = np.reshape(l1.T, (epochs, c1))
x21 = np.reshape(l2.T, (epochs, c2))
x1mean = np.zeros(epochs)
x2mean = np.zeros(epochs)
for i in range(epochs):
    x1mean[i] = np.mean(x11[i, :])
    x2mean[i] = np.mean(x21[i, :])

def generateNumber(num):
    mylist = []
    for i in range(num):
        mylist.append(i)
    return mylist

xp = np.array(generateNumber(epochs))
plt.plot(xp, x1mean, 'm*', label='Train_loss')
plt.plot(xp, x2mean, 'ks', label='Test_loss')
plt.xlabel('Num_epochs')
plt.ylabel('Train and Test Loss ')
plt.title('Train and Test Loss!!!')
plt.legend(loc='best', frameon=False)
plt.savefig('CIFAR_figure2.png')

plt.figure(4)
l3 = np.array(train_acc_list)
l4 = np.array(test_acc_list)
xl = np.arange(0, epochs, 1)
plt.plot(xp, l3, 'b*', label='Train_accuracy')
plt.plot(xp, l4, 'rs', label='Test_accuracy')
plt.xlabel('Num_epochs')
plt.ylabel('Train and Test accuracy ')
plt.title('Train and Test accuracy')
plt.legend(loc='lower right', frameon=False)
plt.savefig('CIFAR_figure3.png')        

tx2=time.time()

print('execution time is:', tx2-tx1)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')