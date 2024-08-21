import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2


factors = [1, 1, 1, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]


class PixelNorm(nn.Module):
    '''Pixel Normalization'''
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(
            torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon
        )


class WSConv2d(nn.Module):
    '''Convolutional layer with weight scaling'''
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.scale = (gain / (self.conv.weight[0].numel())) ** 0.5  # constant
        
        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # print('self.scale:', self.scale)  
        return self.conv(x * self.scale)
    

class ConvBlock(nn.Module):
    '''a Convolutional Block with Convolutional layers, Activation layers and Normalization layers'''
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        if self.pn:
            x = self.pn(x) 
        x = self.leaky(self.conv2(x))
        if self.pn:
            x = self.pn(x)
        return x


class Generator(nn.Module):
    '''
    the Generator of GAN
    parameters:
        z_dim: size of input dimension,
        in_channels: number of output channels of the deconvolutional layer,
        img_size: the size of the output image,
        img_channels: the number of channels of the output image
    '''
    def __init__(self, z_dim: int=512, in_channels: int=512, img_size: int=1024, img_channels: int=3):
        super(Generator, self).__init__()
        
        # initialize layer, transform vector to a feature map
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        # progression blocks and rgb layers
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        channels = in_channels
        for idx in range(int(log2(img_size/4)) + 1):
            conv_in = channels 
            conv_out = int(in_channels * factors[idx])

            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(WSConv2d(conv_out, img_channels, kernel_size=1, stride=1, padding=0))

            channels = conv_out

    def fade_in(self, alpha, upscaled, generated):
        assert 0 <= alpha <= 1, "Alpha not between 0 and 1"
        assert upscaled.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        upscaled = self.initial(x)      
        out = self.prog_blocks[0](upscaled) 

        if steps == 0:
            return self.rgb_layers[0](out)
        
        for step in range(1, steps+1):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)                
        return self.fade_in(alpha, final_upscaled, final_out)
    

class Discriminator(nn.Module):
    '''
    the Generator of GAN
    parameters:
        z_dim: size of input dimension,
        in_channels: number of output channels of the deconvolutional layer,
        img_size: the size of the input image,
        img_channels: the number of channels of the input image
    '''
    def __init__(self, img_size=1024, z_dim=512, in_channels=512, img_channels=3):
        super(Discriminator, self).__init__()

        # progression blocks and rgb layers
        channels = in_channels
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        for idx in range(int(log2(img_size/4)) + 1):
            conv_in = int(in_channels * factors[idx])
            conv_out = channels
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            channels = conv_in

        # for down sampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # output layer
        self.conv_get = WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv = WSConv2d(in_channels, z_dim, kernel_size=4, stride=1, padding=0)
        self.linear = nn.Linear(z_dim, 1)

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avgpooling and output from CNN"""
        assert 0 <= alpha <= 1, "Alpha needs to be between [0, 1]"
        assert downscaled.shape == out.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        out = self.rgb_layers[steps](x) 
        
        if steps == 0: # image is 4x4
            out = self.minibatch_std(out)
            out = self.conv_get(out)
            out = self.conv(out)
            return self.linear(out.view(-1, out.shape[1]))
        
        downscaled = self.rgb_layers[steps - 1](self.avg_pool(x))
        out = self.avg_pool(self.prog_blocks[steps](out))
        out = self.fade_in(alpha, downscaled, out)

        # start from the steps-1 layer
        for step in range(steps-1, 0, -1):
            downscaled = self.avg_pool(out)
            out = self.prog_blocks[step](downscaled)

        
        # the last layer
        out = self.minibatch_std(out)
        out = self.conv_get(out)
        out = self.conv(out)
        return self.linear(out.view(-1, out.shape[1]))

