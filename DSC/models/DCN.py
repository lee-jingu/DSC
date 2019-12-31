import torch
import torch.nn as nn

from .layers import *
from utils.imutils import gaussian

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        dskip = []
        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=2, padding=1, bias=True))
        #self.add_module('BN1', nn.BatchNorm2d(in_channels))
        
        cur_channels_count = out_chans_first_conv
        #print('1', cur_channels_count)

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        self.dilatedConv = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            #print('21', cur_channels_count)
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
            #print('22', skip_connection_channel_counts)
        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels
        #print('3', cur_channels_count)
        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels
            #print('4', cur_channels_count)
        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        #print('5', cur_channels_count)
        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]
        #print('6', cur_channels_count)
        ## Softmax ##
        out_channels_count = 68
        
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=out_channels_count, kernel_size=1, stride=2,
                   padding=0, bias=True)

        '''
        self.BatchNorm = nn.Sequential(
            nn.Conv2d(cur_channels_count, n_classes, kernel_size= 3, stride = 1, bias= True , padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(True))  

        '''
        
        self.dilations = [1,1,2,4,8,16]
        #print('cur_channels_count', cur_channels_count)
        for i, d in enumerate(self.dilations):
            dskip.append(out_channels_count)
            #print('dskip', dskip) 

            self.dilatedConv.append(
                self.dilated(in_channels=out_channels_count, out_channels=68, kernel_size=3, stride=1, padding = d,dilation=d))
            
            out_channels_count = dskip.pop()
            #print('out0', out_channels_count)
            out_channels_count = out_channels_count + 68
            
            #print('out',out_channels_count)
            
            #self.convv = nn.Conv2d(136, n_classes, 3, stride = 1,bias =True, padding=1)
        
        self.convv1 = nn.Conv2d(out_channels_count, n_classes, 3 ,stride = 1, bias =True, padding=1)
        

    def dilated(self, in_channels, out_channels, kernel_size, stride , padding ,dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels= out_channels,kernel_size= kernel_size, 
                stride = stride , padding = padding ,dilation= dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        out = self.firstconv(x)
        #out = self.BN1(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        x = self.finalConv(out)
        #print(x.shape)
        #print('x',x.shape)
        
        #top_side = self.BatchNorm(out)
        kernel_stacked = np.tile(gaussian(size = 45, sigma = 0.01), (68, 1, 1, 1))    # [numLabels x 1 x 45 x 45]
        kernel_stacked = torch.from_numpy(kernel_stacked)
        kernel = Variable(kernel_stacked, requires_grad=False)
        
        o_main = nn.functional.conv2d(x, kernel.cuda(), stride = 1 ,padding = 22,groups=68) # 22 = 45
        #print('o_main', o_main.shape)

        askip = []
        #print('x', x.shape)
        
        for i in range(len(self.dilations)):
            askip.append(x)
            #print(len(askip))
            x = self.dilatedConv[i](x)
            skipp = askip.pop()
            #print('skipp', skipp.shape)
            #print('xx',x.shape)
            x = torch.cat((skipp,x),1)
            #print('xxx',x.shape)
            #x = self.convv(x)
            
        x = self.convv1(x)
        #print("X.out", x.shape)
        #print(x.shape, o_main.shape)
        o = x + o_main

        

        return [o]


def FCDenseNet57(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)

def FCDenseNet67(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)



