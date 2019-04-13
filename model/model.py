import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import NodeOp
from .dag_layer import DAGLayer
from .sep_conv import SeparableConv2d


class RandWire(nn.Module):
    def __init__(self, hp, graphs):
        super(RandWire, self).__init__()
        self.chn = hp.model.channel
        self.cls = hp.model.classes
        self.im = hp.model.input_maps
        # didn't used nn.Sequential for debugging purpose
        # self.conv1 = SeparableConv2d(1, self.chn//2, kernel_size=3, padding=1, stride=2)
        self.conv1 = nn.Conv2d(self.im, self.chn//2, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(self.chn//2)       
        # self.conv2 = SeparableConv2d(self.chn//2, self.chn, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(self.chn//2, self.chn, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(self.chn)
        self.dagly3 = DAGLayer(self.chn, self.chn, graphs[0]['num_nodes'], graphs[0]['edges'])
        self.dagly4 = DAGLayer(self.chn, 2*self.chn, graphs[1]['num_nodes'], graphs[1]['edges'])
        self.dagly5 = DAGLayer(2*self.chn, 4*self.chn, graphs[2]['num_nodes'], graphs[2]['edges'])
        # self.convlast = SeparableConv2d(4*self.chn, 1280, kernel_size=1)
        self.convlast = nn.Conv2d(4*self.chn, 1280, kernel_size=1)
        self.bnlast = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, self.cls)

    def forward(self, y):
        # y: [B, im, 224, 224]
        # conv1
        y = self.conv1(y) # [B, chn//2, 112, 112]
        y = self.bn1(y) # [B, chn//2, 112, 112]

        # conv2
        y = F.relu(y) # [B, chn//2, 112, 112]
        y = self.conv2(y) # [B, chn, 56, 56]
        y = self.bn2(y) # [B, chn, 56, 56]

        # conv3, conv4, conv5
        y = self.dagly3(y) # [B, chn, 28, 28]
        y = self.dagly4(y) # [B, 2*chn, 14, 14]
        y = self.dagly5(y) # [B, 4*chn, 7, 7]

        # classifier
        y = F.relu(y) # [B, 4*chn, 7, 7]
        y = self.convlast(y) # [B, 1280, 7, 7]
        y = self.bnlast(y) # [B, 1280, 7, 7]
        y = F.adaptive_avg_pool2d(y, (1, 1)) # [B, 1280, 1, 1]
        y = y.view(y.size(0), -1) # [B, 1280]
        y = self.fc(y) # [B, cls]
        y = F.log_softmax(y, dim=1) # [B, cls]
        return y
