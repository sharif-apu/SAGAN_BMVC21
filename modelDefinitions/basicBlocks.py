import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torchsummary import summary

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

def conv9x9(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, bias=True)

def swish(x):
    return x * torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class resBlock(nn.Module):
    def __init__(self, in_features):
        super(resBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class pixelShuffleUpsampling(nn.Module):
    def __init__(self, inputFilters, scailingFactor=2):
        super(pixelShuffleUpsampling, self).__init__()
    
        self.upSample = nn.Sequential(  nn.Conv2d(inputFilters, inputFilters * (scailingFactor**2), 3, 1, 1),
                                        nn.BatchNorm2d(inputFilters * (scailingFactor**2)),
                                        nn.PixelShuffle(upscale_factor=scailingFactor),
                                        nn.PReLU()
                                    )
    def forward(self, tensor):
        return self.upSample(tensor)


class sapatialasymmetricAttention(nn.Module):
    def __init__(self, cin, cout, stride=1, padding_mode='zeros'):
        super(sapatialasymmetricAttention, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=cin, out_channels=cout,
                                     kernel_size=(9, 9), stride=stride,
                                     padding=4, bias=True, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=cin, out_channels=cout, 
                                  kernel_size=(3, 1), stride=stride,
                                  padding=(1, 0), bias=True, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=cin, out_channels=cout, 
                                  kernel_size=(1, 3), stride=stride, 
                                  padding=(0, 1), bias=True, padding_mode=padding_mode)

        self.convCat = conv9x9(2,1)


        self.sigmoid = torch.nn.Sigmoid()
        self.depth = SELayer(cout)
        self.coSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.activation = nn.ReLU()
        self.norm = nn.InstanceNorm2d(cout)

    def forward(self, x):
        
        # Global Feature Extraction
        xsquare = self.square_conv(x)
        vertical_outputs = self.ver_conv(x)
        horizontal_outputs = self.hor_conv(x)

        
        # Asymetric Convolution with Sigmoid Gate

        avg_out = torch.mean(vertical_outputs, dim=1, keepdim=True)
        max_out, _ = torch.max(vertical_outputs, dim=1, keepdim=True)
        xvertical_cat = torch.cat([avg_out, max_out], dim=1)
        xvertical = self.sigmoid(self.convCat(xvertical_cat))

        avg_out = torch.mean(horizontal_outputs, dim=1, keepdim=True)
        max_out, _ = torch.max(horizontal_outputs, dim=1, keepdim=True)
        xhorizontal_cat = torch.cat([avg_out, max_out], dim=1)
        xhorizontal = self.sigmoid(self.convCat(xhorizontal_cat))

        # Bi-directional Aggregation 
        x0 = xvertical + xhorizontal  

        # Depth Attention
        x9 = self.depth(xsquare)

        return x9 * x0  

class sapatialasymmetricAttentionModule(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(sapatialasymmetricAttentionModule, self).__init__()
        self.multikernelconv1 = sapatialasymmetricAttention(cin, cout, stride=stride)
        self.activation = nn.LeakyReLU(0.2, inplace = True)
        self.norm = nn.InstanceNorm2d(cout)

    def forward(self, x):        
        return self.activation(self.multikernelconv1(x))

#net = MultiKernelConv2(64,64)
#summary(net, input_size = (64,128, 128))
#print ("reconstruction network")
#net = depthAttentiveResBlock(32, 32,1)
#summary(net, input_size = (32, 128, 128))
#print ("reconstruction network")