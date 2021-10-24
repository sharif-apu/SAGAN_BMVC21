import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    

class attentionNet(nn.Module):
    def __init__(self, squeezeFilters = 64, expandFilters = 64, depth = 1):
        super(attentionNet, self).__init__()

        # Input Block
        self.inputConv = nn.Conv2d(3, squeezeFilters, 3,1,1)
        depthAttenBlock1 = []
        for i in range (depth):
            depthAttenBlock1.append( resBlock(squeezeFilters))
        self.depthAttention1 = nn.Sequential(*depthAttenBlock1)
        self.spatialAttention1 = sapatialasymmetricAttentionModule(squeezeFilters, squeezeFilters)
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1) 

        depthAttenBlock2 = []
        for i in range (depth):
            depthAttenBlock2.append( resBlock(128))
        self.depthAttention2 = nn.Sequential(*depthAttenBlock2)
        self.spatialAttention2 = sapatialasymmetricAttentionModule(128, 128)
        self.down2 = nn.Conv2d(128, 192, 3, 2, 1) 

        depthAttenBlock3 = []
        for i in range (depth):
            depthAttenBlock3.append( resBlock(192))
        self.depthAttention3 = nn.Sequential(*depthAttenBlock3)
        self.spatialAttention3 = sapatialasymmetricAttentionModule(192, 192)
        self.down3 = nn.Conv2d(192, 256, 3, 2, 1) 


        #MID blocks
        depthAttenBlock4_1 = []
        for i in range (depth):
            depthAttenBlock4_1.append( resBlock(256))
        self.depthAttention4_1 = nn.Sequential(*depthAttenBlock4_1)
        self.spatialAttention4_1 = sapatialasymmetricAttentionModule(256, 256)

        depthAttenBlock4_2 = []
        for i in range (depth):
            depthAttenBlock4_2.append( resBlock(256))
        self.depthAttention4_2 = nn.Sequential(*depthAttenBlock4_2)
        self.spatialAttention4_2 = sapatialasymmetricAttentionModule(256, 256)
        


        self.convUP1 = nn.Conv2d(256, 192, 3, 1, 1) 
        self.psUpsampling1 = pixelShuffleUpsampling(inputFilters=192, scailingFactor=2)

        depthAttenBlock5 = []
        for i in range (depth):
            depthAttenBlock5.append( resBlock(192))
        self.depthAttention5 = nn.Sequential(*depthAttenBlock5)
        self.spatialAttention5 = sapatialasymmetricAttentionModule(192, 192)


        self.convUP2 = nn.Conv2d(192, 128, 3, 1, 1) 
        self.psUpsampling2 = pixelShuffleUpsampling(inputFilters=128, scailingFactor=2)
        depthAttenBlock6 = []
        for i in range (depth):
            depthAttenBlock6.append( resBlock(128))
        self.depthAttention6 = nn.Sequential(*depthAttenBlock6)
        self.spatialAttention6 = sapatialasymmetricAttentionModule(128,128)

        self.convUP3 = nn.Conv2d(128, 64, 3, 1, 1) 
        self.psUpsampling3 = pixelShuffleUpsampling(inputFilters=64, scailingFactor=2)
        depthAttenBlock7 = []
        for i in range (depth):
            depthAttenBlock7.append( resBlock(64))
        self.depthAttention7 = nn.Sequential(*depthAttenBlock7)
        self.spatialAttention7 = sapatialasymmetricAttentionModule(64,64)


        self.convOut = nn.Conv2d(squeezeFilters,3,1,)


        # Convolution Gate
        self.gate1 = conv1x1(192, 192)
        self.gate2 = conv1x1(128, 128)
        self.gate3 = conv1x1(64, 64)


        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):

        xInp = F.leaky_relu(self.inputConv(img))

        # Encoder
        xSP1 = self.depthAttention1(xInp)
        xFA1 = F.leaky_relu(self.spatialAttention1(xSP1))
        xDS1 = F.leaky_relu(self.down1(xFA1))

        xSP2 = self.depthAttention2(xDS1)
        xFA2 = self.spatialAttention2(xSP2) 
        xDS2 = F.leaky_relu(self.down2(xFA2))

        xSP3 = self.depthAttention3(xDS2)
        xFA3 = self.spatialAttention3(xSP3)
        xDS3 = F.leaky_relu(self.down3(xFA3))

        # High Level Feature Extraction (MidNet)
        xMID_1 = self.depthAttention4_1(xDS3)
        xMID_1 = self.spatialAttention4_1(xMID_1)

        xMID_2 = self.depthAttention4_1(xMID_1)
        xMID_2 = self.spatialAttention4_1(xMID_2) + xMID_1

        # Decoder
        xCP1 = F.leaky_relu(self.convUP1(xMID_2))
        xPS1 = self.psUpsampling1(xCP1) 
        xSP5 = self.depthAttention5(xPS1)
        xFA5 = self.spatialAttention5(xSP5) + self.gate1(xFA3)

        xCP2 = F.leaky_relu(self.convUP2(xFA5))
        xPS2 = self.psUpsampling2(xCP2) 
        xSP6 = self.depthAttention6(xPS2)
        xFA6 = self.spatialAttention6(xSP6) + self.gate2(xFA2)

        xCP3 = F.leaky_relu(self.convUP3(xFA6))
        xPS3 = self.psUpsampling3(xCP3) 
        xSP7 = self.depthAttention7(xPS3)
        xFA7 = self.spatialAttention7(xSP7) + self.gate3(xFA1)
        
        return torch.tanh(self.convOut(xFA7) + img)
        
        
    
    def _initialize_weights(self):

        self.inputConv.apply(init_weights)
        self.depthAttention1.apply(init_weights)
        self.spatialAttention1.apply(init_weights)
        
        self.down1.apply(init_weights)
        self.depthAttention2.apply(init_weights)
        self.spatialAttention2.apply(init_weights)
        
        self.down2.apply(init_weights)
        self.depthAttention3.apply(init_weights)
        self.spatialAttention3.apply(init_weights)
        
        self.convUP1.apply(init_weights)
        self.psUpsampling1.apply(init_weights)
        self.depthAttention4.apply(init_weights)
        self.spatialAttention4.apply(init_weights)
       
        self.convUP2.apply(init_weights)
        self.psUpsampling2.apply(init_weights)
        self.depthAttention5.apply(init_weights)
        self.spatialAttention5.apply(init_weights)
        
        self.convOut.apply(init_weights)

#net = attentionNet()
#summary(net, input_size = (3, 128, 128))
#print ("reconstruction network")