# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F

# from switchable_BN import *

class UNet_online(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=3,    # 2 ** (wf  + i)   # else 10 * (i + 1)
        kernel_size = 5,
        padding= int((5 - 1)/2),
        batch_norm=False,
        batch_norm_switchable=False,
        up_mode='upconv',   # if want to switch to 'upsample' need to do bicubic!!!
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet_online, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.kernel_size = kernel_size
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf  + i), padding, batch_norm, batch_norm_switchable, kernel_size)
            )
            prev_channels = 2 ** (wf  + i)
            
        # self.up_path = nn.ModuleList()
        # for i in reversed(range(depth - 1)):
        #     self.up_path.append(
        #         UNetUpBlock(prev_channels, 2 ** (wf  + i), up_mode, padding, batch_norm, batch_norm_switchable, kernel_size)
        #     )
        #     prev_channels = 2 ** (wf  + i)

        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1) ### size of feature map is 16 x 16 x 2 with 2 classes (i.e. 2 filters)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)   ### makes it into feature map of size 1 x 1 x 1, with 2 filters (i.e. 2 classes)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool3d(x, 2)

        # for i, up in enumerate(self.up_path):
        #     x = up(x, blocks[-i - 1])

        return self.avg_pool(self.last(x))


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, batch_norm_switchable, kernel_size):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size, padding=int(padding)))   # can be changed to 5
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))
        elif batch_norm_switchable:
            block.append(SwitchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=kernel_size, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))
        elif batch_norm_switchable:
            block.append(SwitchNorm3d(out_size))


        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


# class UNetUpBlock(nn.Module):
#     def __init__(self, in_size, out_size, up_mode, padding, batch_norm, batch_norm_switchable, kernel_size):
#         super(UNetUpBlock, self).__init__()
#         if up_mode == 'upconv':
#             self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, output_padding = 0)
#         elif up_mode == 'upsample':
#             self.up = nn.Sequential(
#                 nn.Upsample(mode='trilinear', scale_factor=2),
#                 nn.Conv3d(in_size, out_size, kernel_size=1),
#             )

#         self.conv_block = UNetConvBlock(out_size * 2, out_size, padding, batch_norm, batch_norm_switchable, kernel_size)

#     def center_crop(self, layer, target_size):
#         _, _, layer_depth, layer_height, layer_width = layer.size()
#         diff_y = (layer_height - target_size[1]) // 2
#         diff_x = (layer_width - target_size[2]) // 2
#         diff_z = (layer_depth - target_size[0]) // 2
#         return layer[
#             :, :, diff_z: (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])
#         ]

#     def forward(self, x, bridge):
#         up = self.up(x)
#         crop1 = self.center_crop(bridge, up.shape[2:])
#         out = torch.cat([up, crop1], 1)
#         out = self.conv_block(out)

#         return out