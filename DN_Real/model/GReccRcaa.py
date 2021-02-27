import model.common as common
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from model.merge_unit import merge_block

class RR2(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RR2, self).__init__()
        n_resblocks = 16
        n_feats = 64  
        kernel_size = 3

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        msa = CES(in_channels=n_feats,num=6,blocks=3)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(msa)
        for i in range(n_resblocks // 2):
            m_body.append(common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        res = self.head(x)

        res = self.body(res)

        res = self.tail(res)

        return x+res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class CES(nn.Module):
    def __init__(self,in_channels,num=4,blocks=2):
        super(CES,self).__init__()
        print('num_RB:',num)
        print('num_blocks',blocks)
        self.blocks = blocks
        RBS1 = [
            common.ResBlock(
                common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num//2)
        ]
        RBS2 = [
            common.ResBlock(
                common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
            ) for _ in range(num//2)
        ]
        self.RBS1 = nn.Sequential(
            *RBS1
        )
        self.RBS2 = nn.Sequential(
            *RBS2
        )
        self.c1 = merge_block(in_channels = in_channels,out_channels=in_channels)
        self.c2 = merge_block(in_channels = in_channels,out_channels=in_channels)
        self.c3 = merge_block(in_channels = in_channels,out_channels=in_channels)
    def forward(self, x):
        out = self.c1(x)
        out = self.RBS1(out)
        out = self.c2(out)
        out = self.RBS2(out)
        out = self.c3(out)
        return out
"""
fundamental functions
"""
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings