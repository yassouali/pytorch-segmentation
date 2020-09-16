from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from itertools import chain

''' 
-> Dense upsampling convolution block
'''

class DUC(nn.Module):
    def __init__(self, in_channels, out_channles, upscale):
        super(DUC, self).__init__()
        out_channles = out_channles * (upscale ** 2)
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)

        initialize_weights(self)
        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        '''
        Even with pixel shuffle we still have check board artifacts,
        the solution is to initialize the d**2 feature maps with the same
        radom weights: https://arxiv.org/pdf/1707.02937.pdf
        '''
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

''' 
-> ResNet BackBone
'''

class ResNet_HDC_DUC(nn.Module):
    def __init__(self, in_channels, output_stride, pretrained=True, dilation_bigger=False):
        super(ResNet_HDC_DUC, self).__init__()

        model = models.resnet101(pretrained=pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 4: list(self.layer0.children())[0].stride = (1, 1)

        d_res4b = []
        if dilation_bigger:
            d_res4b.extend([1, 2, 5, 9]*5 + [1, 2, 5])
            d_res5b = [5, 9, 17]
        else:
            # Dialtion-RF
            d_res4b.extend([1, 2, 3]*7 + [2, 2])
            d_res5b = [3, 4, 5]

        l_index = 0
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                d = d_res4b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        l_index = 0
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                d = d_res5b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features

''' 
-> The Atrous Spatial Pyramid Pooling
'''

def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride, assp_channels=6):
        super(ASSP, self).__init__()

        assert output_stride in [4, 8], 'Only output strides of 8 or 16 are suported'
        assert assp_channels in [4, 6], 'Number of suported ASSP branches are 4 or 6'
        dilations = [1, 6, 12, 18, 24, 36]
        dilations = dilations[:assp_channels]
        self.assp_channels = assp_channels

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])
        if self.assp_channels == 6:
            self.aspp5 = assp_branch(in_channels, 256, 3, dilation=dilations[4])
            self.aspp6 = assp_branch(in_channels, 256, 3, dilation=dilations[5])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256*(self.assp_channels + 1), 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        if self.assp_channels == 6:
            x5 = self.aspp5(x)
            x6 = self.aspp6(x)
        x_avg_pool = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        if self.assp_channels == 6:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x5, x6, x_avg_pool), dim=1))
        else:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x_avg_pool), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

''' 
-> Decoder
'''

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.DUC = DUC(256, 256, upscale=2)

        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        x = self.DUC(x)
        if x.size() != low_level_features.size():
            # One pixel added with a conv with stride 2 when the input size in odd
            x = x[:, :, :low_level_features.size(2), :low_level_features.size(3)]
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

'''
-> Deeplab V3 + with DUC & HDC
'''

class DeepLab_DUC_HDC(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, output_stride=8, freeze_bn=False, **_):
        super(DeepLab_DUC_HDC, self).__init__()

        self.backbone = ResNet_HDC_DUC(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
        low_level_channels = 256

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)
        self.DUC_out = DUC(num_classes, num_classes, 4)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = self.DUC_out(x)
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters(), self.DUC_out.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

