"""
Dense Convolutional Encoder-Decoder Networks

Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from load_data import load_data
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
#additional ---->

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#--->
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
                            kernel_size=5, stride=2, padding=2, 
                            output_padding=1, bias=False, drop_rate=drop_rate)
        self.features.add_module('LastTransUp', last_trans_up)
        if out_activation is not 'None':
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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device='cuda:0'

torch.manual_seed(args.seed)

#device = torch.device("cuda:0" if args.cuda else "cpu")
batch_size =16

#class Flatten(nn.Module):
#    def forward(self, input):
#        return input.view(input.size(0), -1)
#
#class UnFlatten(nn.Module):
#    def forward(self, input, size=1024):
#        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=3072, z_dim=6400):
        super(VAE, self).__init__()
        self.encoder = Encoder(3, 48, blocks=(3,), growth_rate=16, 
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
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        #print ('here',h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        #print('>>>>z size',z.size())
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        #print('>>>>>>>>s',h.size())

        #h = h.view(-1,18432)
        z, mu, logvar = self.bottleneck(h.view(-1,3072))
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        #print('dec size ---->>>>>',z.size())
        z = z.view(16,48,8,8)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar







#changed here ------------->
#batch_size = 10

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
model = VAE(image_channels=3).cuda()  #new  one
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4225), size_average=False)
    BCE = F.mse_loss(recon_x.view(-1,3072), x.view(-1,3072), size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        #print (data)
        recon_batch, mu, logvar = model(data)
        #print('reco image',recon_batch.size())
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('Hello ')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_error = train_loss / len(train_loader.dataset)
    np.savetxt('Train_loss=%d.txt'%epoch, np.array(train_error).reshape(1,), fmt='%.4f',delimiter=',')
    scheduler.step(train_loss)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            input_data = data.view(-1,3072)
            recon_batch_data = recon_batch.view(-1,3072)
            np.savetxt('T_Drop_Layer_results/input=%d.txt'%epoch, input_data.data, delimiter=',')  
            np.savetxt('T_Drop_Layer_results/output=%d.txt'%epoch, recon_batch_data.data, delimiter=',')

#            if i == 0:
#
#                img_data = data.view(16,3,32,32).data
#                vmax=np.amax(img_data.numpy())
#                vmin=np.amin(img_data.numpy())
#                levels = np.linspace(vmin,vmax,100)
#                #plt1 = plt.figure()
#                #plt2 = plt.figure()
#                plt.figure(figsize=(8,6))
#                A = plt.contourf(img_data.numpy()[0,0])
#                plt.colorbar()
#                    #plt.show()
#                plt.savefig('T_Drop_Layer_results/data_'+ str(epoch) +'.png') 
#                img_recon = (recon_batch.view(16, 3, 32, 32).data)
#                plt.figure(figsize=(8,6))
#                B = plt.contourf(img_recon.numpy()[0,0])
#                plt.colorbar()
#                    #plt.show()
#                plt.savefig('T_Drop_Layer_results/recon_'+ str(epoch) +'.png') 
#
#
#                #n = min(data.size(0), 8)
#                #comparison = torch.cat([data[:n],
#                                      #recon_batch.view(10, 1, 65, 65)[:n]])
#
#
#                #save_image(comparison.cpu(),
#                         #'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    np.savetxt('Test_loss=%d.txt'%epoch, np.array(test_loss).reshape(1,), fmt='%.4f',delimiter=',')
epochs = 100
for epoch in range(1, epochs + 1):
    train(epoch)
    with torch.no_grad():
        test(epoch)
print('END OF THE PROGRAM')
        
torch.save(model, 'VAE_var_test1_6400.pt')