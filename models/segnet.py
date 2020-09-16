from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from math import ceil

class SegNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained= pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                    self.stage4_decoder, self.stage5_decoder)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



class DecoderBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.ConvTranspose2d(inchannels//4, inchannels//4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(inchannels//2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LastBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False),
                nn.BatchNorm2d(inchannels//4))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SegResNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)

        # Decoder
        resnet50_untrained = models.resnet50(pretrained=False)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.first_conv, self.encoder], False)

    def forward(self, x):
        inputsize = x.size()

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2]-(h_diff-1), w_diff: x.size()[3]-(w_diff-1)]
        else:
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)
        
        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


