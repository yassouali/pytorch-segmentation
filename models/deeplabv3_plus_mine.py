# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:16:18 2021

@author: Administrator
"""

from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights,set_trainable
from itertools import chain
'''
'xception_65.pth'URL:https://github.com/zhangtianlun12/deeplabv3-/releases/download/v0.1/xception_65.pth
'''


''' 
-> ResNet BackBone
'''

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
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

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)
        
        if output_stride == 8: 
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


"""
Created on Fri Sep 13 19:04:23 2019

@author: shirhe-lyh


Implementation of Xception model.
Xception: Deep Learning with Depthwise Separable Convolutions, F. Chollect,
    arxiv:1610.02357 (https://arxiv.org/abs/1610.02357).

Official tensorflow implementation:
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
"""

import collections
import os
import torch


_DEFAULT_MULTI_GRID = [1, 1, 1]
# The cap for torch.clamp
_CLIP_CAP = 6
_BATCH_NORM_PARAMS = {
    'eps': 0.001,
    'momentum': 0.9997,
    'affine': True,
}


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing an Xception block.
    
    Its parts are:
        scope: The scope of the block.
        unit_fn: The Xception unit function which takes as input a tensor and
            returns another tensor with the output of the Xception unit.
        args: A list of length equal to the number of units in the block. The
            list contains one dictionary for each unit in the block to serve 
            as argument to unit_fn.
    """
    
    
def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d 
            operation. Should be a positive integer.
        rate: An integer, rate for atrous convolution.
        
    Returns:
        padded_inputs: A tensor of size [batch, height_out, width_out, 
            channels] with the input, either intact (if kernel_size == 1) or 
            padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = torch.nn.functional.pad(
        inputs, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Conv2dSame(torch.nn.Module):
    """Strided 2-D convolution with 'SAME' padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, rate=1):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
        """
        super(Conv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1
        if self._without_padding:
            # Here, we assume that floor(padding) = padding
            padding = (kernel_size - 1) * rate // 2
            self._conv = torch.nn.Conv2d(in_channels, 
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         dilation=rate,
                                         padding=padding,
                                         bias=False)
        else:
            self._conv = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         dilation=rate,
                                         bias=False)
        self._batch_norm = torch.nn.BatchNorm2d(out_channels, 
                                                **_BATCH_NORM_PARAMS)
        self._relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class SeparableConv2dSame(torch.nn.Module):
    """Strided 2-D separable convolution with 'SAME' padding."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 depth_multiplier, stride, rate, use_explicit_padding=True, 
                 activation_fn=None, regularize_depthwise=False, **kwargs):
        """Constructor.
        
        If stride > 1 and use_explicit_padding is True, then we do explicit
        zero-padding, followed by conv2d with 'VALID' padding.
        
        Args:
            in_channels: An integer, the number of input filters.
            out_channels: An integer, the number of output filters.
            kernel_size: An integer with the kernel_size of the filters.
            depth_multiplier: The number of depthwise convolution output
                channels for each input channel. The total number of depthwise
                convolution output channels will be equal to `num_filters_in *
                depth_multiplier`.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
            activation_fn: Activation function.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            **kwargs: Additional keyword arguments to pass to torch.nn.Conv2d.
        """
        super(SeparableConv2dSame, self).__init__()
        self._kernel_size = kernel_size
        self._rate = rate
        self._without_padding = stride == 1 or not use_explicit_padding
        
        out_channels_depthwise = in_channels * depth_multiplier
        if self._without_padding:
            # Separable convolution for padding 'SAME'
            # Here, we assume that floor(padding) = padding
            padding = (kernel_size - 1) * rate // 2
            self._conv_depthwise = torch.nn.Conv2d(in_channels, 
                                                   out_channels_depthwise,
                                                   kernel_size=kernel_size, 
                                                   stride=stride, 
                                                   dilation=rate,
                                                   groups=in_channels,
                                                   padding=padding,
                                                   bias=False,
                                                   **kwargs)
        else:
            # Separable convolution for padding 'VALID'
            self._conv_depthwise = torch.nn.Conv2d(in_channels,
                                                   out_channels_depthwise,
                                                   kernel_size=kernel_size, 
                                                   stride=stride,
                                                   dilation=rate,
                                                   groups=in_channels,
                                                   bias=False,
                                                   **kwargs)
        self._batch_norm_depthwise = torch.nn.BatchNorm2d(
            out_channels_depthwise, **_BATCH_NORM_PARAMS)
        self._conv_pointwise = torch.nn.Conv2d(out_channels_depthwise,
                                               out_channels,
                                               kernel_size=1, 
                                               stride=1,
                                               bias=False,
                                               **kwargs)
        self._batch_norm_pointwise = torch.nn.BatchNorm2d(
            out_channels, **_BATCH_NORM_PARAMS)
        self._activation_fn = activation_fn
    
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height_in, width_in, channels].
        
        Returns:
            A 4-D tensor of size [batch, height_out, width_out, channels] with 
                the convolution output.
        """
        if not self._without_padding:
            x = fixed_padding(x, self._kernel_size, self._rate)
        x = self._conv_depthwise(x)
        x = self._batch_norm_depthwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        x = self._conv_pointwise(x)
        x = self._batch_norm_pointwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
    

class XceptionModule(torch.nn.Module):
    """An Xception module.
    
    The output of one Xception module is equal to the sum of `residual` and
    `shortcut`, where `residual` is the feature computed by three seperable
    convolution. The `shortcut` is the feature computed by 1x1 convolution
    with or without striding. In some cases, the `shortcut` path could be a
    simple identity function or none (i.e, no shortcut).
    """
    
    def __init__(self, in_channels, depth_list, skip_connection_type, stride, 
                 unit_rate_list, rate=1, activation_fn_in_separable_conv=False, 
                 regularize_depthwise=False, use_bounded_activation=False,
                 use_explicit_padding=True):
        """Constructor.
        
        Args:
            in_channels: An integer, the number of input filters.
            depth_list: A list of three integers specifying the depth values
                of one Xception module.
            skip_connection_type: Skip connection type for the residual path.
                Only supports 'conv', 'sum', or 'none'.
            stride: The block unit's stride. Detemines the amount of 
                downsampling of the units output compared to its input.
            unit_rate_list: A list of three integers, determining the unit 
                rate for each separable convolution in the Xception module.
            rate: An integer, rate for atrous convolution.
            activation_fn_in_separable_conv: Includes activation function in
                the seperable convolution or not.
            regularize_depthwise: Whether or not apply L2-norm regularization
                on the depthwise convolution weights.
            use_bounded_activation: Whether or not to use bounded activations.
                Bounded activations better lend themselves to quantized 
                inference.
            use_explicit_padding: If True, use explicit padding to make the
                model fully compatible with the open source version, otherwise
                use the nattive Pytorch 'SAME' padding.
                
        Raises:
            ValueError: If depth_list and unit_rate_list do not contain three
                integers, or if stride != 1 for the third seperable convolution
                operation in the residual path, or unsupported skip connection
                type.
        """
        super(XceptionModule, self).__init__()
        
        if len(depth_list) != 3:
            raise ValueError('Expect three elements in `depth_list`.')
        if len(unit_rate_list) != 3:
            raise ValueError('Expect three elements in `unit_rate_list`.')
        if skip_connection_type not in ['conv', 'sum', 'none']:
            raise ValueError('Unsupported skip connection type.')
            
        # Activation function
        self._input_activation_fn = None
        if activation_fn_in_separable_conv:
            activation_fn = (torch.nn.ReLU6(inplace=False) if 
                             use_bounded_activation else 
                             torch.nn.ReLU(inplace=False))
        else:
            if use_bounded_activation:
                # When use_bounded_activation is True, we clip the feature
                # values and apply relu6 for activation.
                activation_fn = lambda x: torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
                self._input_activation_fn = torch.nn.ReLU6(inplace=False)
            else:
                # Original network design.
                activation_fn = None
                self._input_activation_fn = torch.nn.ReLU(inplace=False)
        self._use_bounded_activation = use_bounded_activation
        self._output_activation_fn = None
        if use_bounded_activation:
            self._output_activation_fn = torch.nn.ReLU6(inplace=True)
         
        # Separable conv block.
        layers = []
        in_channels_ = in_channels
        for i in range(3):
            if self._input_activation_fn is not None:
                layers += [self._input_activation_fn]
            layers += [
                SeparableConv2dSame(in_channels_,
                                    depth_list[i],
                                    kernel_size=3,
                                    depth_multiplier=1,
                                    regularize_depthwise=regularize_depthwise,
                                    rate=rate*unit_rate_list[i],
                                    stride=stride if i==2 else 1,
                                    activation_fn=activation_fn,
                                    use_explicit_padding=use_explicit_padding)]
            in_channels_ = depth_list[i]
        self._separable_conv_block = torch.nn.Sequential(*layers)
        
        # Skip connection
        self._skip_connection_type = skip_connection_type
        if skip_connection_type == 'conv':
            self._conv_skip_connection = torch.nn.Conv2d(in_channels,
                                                         depth_list[-1],
                                                         kernel_size=1,
                                                         stride=stride)
            self._batch_norm_shortcut = torch.nn.BatchNorm2d(
                depth_list[-1], **_BATCH_NORM_PARAMS)
            
    def forward(self, x):
        """
        Args:
            x: A 4-D tensor with shape [batch, height, width, channels].
        
        Returns:
            The Xception module's output.
        """
        residual = self._separable_conv_block(x)
        if self._skip_connection_type == 'conv':
            shortcut = self._conv_skip_connection(x)
            shortcut = self._batch_norm_shortcut(shortcut)
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                shortcut = torch.clamp(shortcut, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + shortcut
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        elif self._skip_connection_type == 'sum':
            if self._use_bounded_activation:
                residual = torch.clamp(residual, -_CLIP_CAP, _CLIP_CAP)
                x = torch.clamp(x, -_CLIP_CAP, _CLIP_CAP)
            outputs = residual + x
            if self._use_bounded_activation:
                outputs = self._output_activation_fn(outputs)
        else:
            outputs = residual
        return outputs
    
    
class StackBlocksDense(torch.nn.Module):
    """Stacks Xception blocks and controls output feature density.
    
    This class allows the user to explicitly control the output stride, which
    is the ratio of the input to output spatial resolution. This is useful for
    dense prediction tasks such as semantic segmentation or object detection.
    
    Control of the output feature density is implemented by atrous convolution.
    """
    
    def __init__(self, blocks, output_stride=None):
        """Constructor.
        
        Args:
            blocks: A list of length equal to the number of Xception blocks.
                Each element is an Xception Block object describing the units
                in the block.
            output_stride: If None, then the output will be computed at the
                nominal network stride. If output_stride is not None, it 
                specifies the requested ratio of input to output spatial
                resolution, which needs to be equal to the product of unit
                strides from the start up to some level of Xception. For
                example, if the Xception employs units with strides 1, 2, 1,
                3, 4, 1, then valid values for the output_stride are 1, 2, 6,
                24 or None (which is equivalent to output_stride=24).
                
        Raises:
            ValueError: If the target output_stride is not valid.
        """
        super(StackBlocksDense, self).__init__()
        
        # The current_stride variable keeps track of the effective stride of
        # the activations. This allows us to invoke atrous convolution whenever
        # applying the next residual unit would result in the activations 
        # having stride larger than the target output_stride.
        current_stride = 1
        
        # The atrous convolution rate parameter.
        rate = 1
        
        layers = []
        for block in blocks:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be '
                                     'reached.')
                # If we have reached the target output_stride, then we need to
                # employ atrous convolution with stride=1 and multiply the
                # atrous rate by the current unit's stride for use subsequent
                # layers.
                if output_stride is not None and current_stride == output_stride:
                    layers += [block.unit_fn(rate=rate, **dict(unit, stride=1))]
                    rate *= unit.get('stride', 1)
                else:
                    layers += [block.unit_fn(rate=1, **unit)]
                    current_stride *= unit.get('stride', 1)
        
        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target ouput_stride cannot be reached.')
            
        self._blocks = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch, height, widht, channels].
            
        Returns:
            Output tensor with stride equal to the specified output_stride.
        """
        x = self._blocks(x)
        return x
    
    
class Xception(torch.nn.Module):
    """Generator for Xception models.
    
    This class generates a family of Xception models. See the xception_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce Xception of various depths.
    """
    
    def __init__(self, blocks, num_classes=None, global_pool=True, 
                 keep_prob=0.5, output_stride=None, scope=None):
        """Constructor.
        
        Args:
            blocks: A list of length equal to the number of Xception blocks.
                Each element is an Xception Block object describing the units
                in the block.
            num_classes: Number of predicted classes for classification tasks.
                If 0 or None, we return the features before the logit layer.
            global_pool: If True, we perform global average pooling before
                computing logits. Set to True for image classification, False
                for dense prediction.
            keep_prob: Keep probability used in the pre-logits dropout layer.
            output_stride: If None, the the output will be computed at the 
                nominal network stride. If output_stride is not None, it
                specifies the requested ratio of input to output spatial
                resolution.
            scope: Optional variable_scope.
                
        Raises:
            ValueError: If the target output_stride is not valid.
        """
        super(Xception, self).__init__()
        
        self._scope = scope
        
        layers = []
        if output_stride is not None:
            if output_stride % 2 != 0:
                raise ValueError('The output_stride must be a multiple of 2.')
            output_stride /= 2
        # Root block function operated on inputs
        layers += [Conv2dSame(3, 32, 3, stride=2),
                   Conv2dSame(32, 64, 3, stride=1)]
        
        # Extract features for entry_flow, middle_flow, and exit_flow
        layers += [StackBlocksDense(blocks, output_stride)]
        
        if global_pool:
            # Global average pooling
            layers += [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))]
        if num_classes:
            layers += [torch.nn.Dropout2d(p=keep_prob, inplace=True),
                       torch.nn.Conv2d(blocks[-1].args[-1]['depth_list'][-1], 
                                       num_classes, 1)]
        self._layers = torch.nn.Sequential(*layers)
                       
    def forward(self, x):
        """
        Args:
            x: A tensor of shape [batch, height, widht, channels].
            
        Returns:
            Output tensor with stride equal to the specified output_stride.
        """
        output = self._layers(x)
        
        
        x1 = self._layers[0](x)
        x2 = self._layers[1](x1)
        low_level_features = self._layers[2]._blocks[0](x2)
        
        #low_level_features = self._layers[2]._blocks[0](x1)
        
        #print('x1',x1.size())
        #print('x2',x2.size())        
        #print('low_level_features',low_level_features.size())
        '''
        if output_stride = None:
            output.size() torch.Size([2, 2048, 7, 7])
            low_level_features.size() torch.Size([2, 128, 56, 56])
        elif output_stride = 16:
            output.size() torch.Size([2, 2048, 14, 14])
            low_level_features.size() torch.Size([2, 128, 56, 56])
        
        
        '''
        
        
        return output,low_level_features
    
    @property
    def scope(self):
        return self._scope
    
    
def xception_block(scope,
                   in_channels,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
    """Helper function for creating a Xception block.
    
    Args:
        scope: The scope of the block.
        in_channels: The number of input filters.
        depth_list: The depth of the bottleneck layer for each unit.
        skip_connection_type: Skip connection type for the residual path. Only
            supports 'conv', 'sum', or 'none'.
        activation_fn_in_separable_conv: Includes activation function in the
            separable convolution or not.
        regularize_depthwise: Whether or not apply L2-norm regularization on 
            the depthwise convolution weights.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last
            unit. All other units have stride=1.
        unit_rate_list: A list of three integers, determining the unit rate in
            the corresponding xception block.
            
    Returns:
        An xception block.
    """
    if unit_rate_list is None:
        unit_rate_list = _DEFAULT_MULTI_GRID
    return Block(scope, XceptionModule, [{
            'in_channels': in_channels,
            'depth_list': depth_list,
            'skip_connection_type': skip_connection_type,
            'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
            'regularize_depthwise': regularize_depthwise,
            'stride': stride,
            'unit_rate_list': unit_rate_list,
            }] * num_units)
    
    

def Xception41(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_41'):
    """Xception-41 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=8,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)
    
    
def xception_41(num_classes=None,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_41',
                pretrained=True,
                checkpoint_path='./pretrained/xception_41.pth'):
    """Xception-41 model."""
    xception = Xception41(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path)
    return xception


def Xception65(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_65'):
    """Xception-65 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)


def xception_65(num_classes=None,
                global_pool=False,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_65',
                pretrained=True,
                checkpoint_path='./pretrained/xception_65.pth'):
    """Xception-65 model."""
    xception = Xception65(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path='./pretrained/xception_65.pth')
    return xception


def Xception71(num_classes=None,
               global_pool=True,
               keep_prob=0.5,
               output_stride=None,
               regularize_depthwise=False,
               multi_grid=None,
               scope='xception_71'):
    """Xception-71 model."""
    blocks = [
        xception_block('entry_flow/block1',
                       in_channels=64,
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       in_channels=128,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block3',
                       in_channels=256,
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block4',
                       in_channels=256,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block5',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       in_channels=728,
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       in_channels=728,
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       in_channels=1024,
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return Xception(blocks=blocks, num_classes=num_classes,
                    global_pool=global_pool, keep_prob=keep_prob,
                    output_stride=output_stride, scope=scope)


def xception_71(num_classes=None,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                scope='xception_71',
                pretrained=True,
                checkpoint_path='./pretrained/xception_71.pth'):
    """Xception-71 model."""
    xception = Xception71(num_classes=num_classes, global_pool=global_pool, 
                          keep_prob=keep_prob, output_stride=output_stride,
                          scope=scope)
    if pretrained:
        _load_state_dict(xception, num_classes, checkpoint_path)
    return xception


def _load_state_dict(model, num_classes, checkpoint_path):
    """Load pretrained weights."""
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        if num_classes is None or num_classes != 1001:
            state_dict.pop('_layers.5.weight')
            state_dict.pop('_layers.5.bias')
        model.load_state_dict(state_dict, strict=False)
        print('Load pretrained weights successfully.')
    else:
        raise ValueError('`checkpoint_path` does not exist.')







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
    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
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

        # Table 2, best performance with two 3x3 convs
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
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

'''
-> Deeplab V3 +
'''

class DeepLab(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True, 
                output_stride=16, freeze_bn=False,freeze_backbone=False, **_):
                
        super(DeepLab, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = xception_65(output_stride=output_stride, pretrained=pretrained,global_pool=False,checkpoint_path='./pretrained/xception_65.pth')
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
                
