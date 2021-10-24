import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import torchvision.models as models
from torchvision.models import vgg19
from modelDefinitions.basicBlocks import *
import torch.nn.init as init
from modelDefinitions.basicBlocks import *    
  

class attentiomDiscriminator(nn.Module):
    def __init__(self):
        super(attentiomDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        # Spatial Tranformar Attention
        self.conv8 = sapatialasymmetricAttentionModule(512, 512, stride=2)
        self.bn8 = nn.BatchNorm2d(512)

        # Transformer Attention

        #self.conv8_1 = sapatialasymmetricAttentionModule(512, 512)
        #self.bn8_1 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        #print(x.shape)
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = self.conv8(x)

        #x = self.conv8_1(x)
        

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


#net = enhanceGenerator(bn=False)
#summary(net, input_size = (3, 128, 128))