import __init__paths
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Any
from utils.trans import to_tsm, to_seg

PARAMS = {
    "input_channels": 1,
    "output_channels": 2,
    "input_transform_fn": lambda x: x / 128. - 1.,
    "input_conv_channels": 32,
    'down_structure': [3, 8, 4],
    'activation_fn': lambda: nn.LeakyReLU(0.1, inplace=True),
    # "normalization_fn": lambda c: nn.GroupNorm(num_groups=4, num_channels=c),
    "normalization_fn": lambda c: nn.BatchNorm2d(num_features=c),
    "drop_rate": 0,
    'growth_rate': 32,
    'bottleneck': 4,
    'compression': 2,
    'use_memonger': True  # ~2.2x memory efficiency (batch_size), 25%~30% slower on GTX 1080.
}


def densenet2d(with_segment=False, snapshot=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)
    model = DenseSharp()
    print("Using DenseSharp model.")
    print(model)
    if snapshot is None:
        initialize(model.modules())
        print("Random initialized.")
    else:
        state_dict = torch.load(snapshot)
        model.load_state_dict(state_dict)
        print("Load weights from `%s`," % snapshot)
    return model


def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            m.bias.data.zero_()


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()

        growth_rate = PARAMS['growth_rate']
        bottleneck = PARAMS['bottleneck']
        activation_fn = PARAMS['activation_fn']
        normalization_fn = PARAMS['normalization_fn']

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.use_memonger = PARAMS['use_memonger']
        self.drop_rate = PARAMS['drop_rate']


        # TODO: consider bias term in conv with GN
        self.add_module('norm_1', normalization_fn(in_channels))
        self.add_module('act_1', activation_fn())
        self.add_module('conv_1', nn.Conv2d(in_channels, bottleneck * growth_rate, kernel_size=1, stride=1,
                                            padding=0, bias=True))

        self.add_module('norm_2', normalization_fn(bottleneck * growth_rate))
        self.add_module('act_2', activation_fn())
        self.add_module('conv_2', nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel_size=3, stride=1,
                                            padding=1, bias=True))

    def forward(self, x):
        super_forward = super(ConvBlock, self).forward
        #x = tsm(x, 32, 'zero')
        if self.use_memonger:
            new_features = checkpoint(super_forward, x)
        else:
            new_features = super_forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    @property
    def out_channels(self):
        return self.in_channels + self.growth_rate


class TransmitBlock(nn.Sequential):
    def __init__(self, in_channels, is_last_block):
        super(TransmitBlock, self).__init__()

        activation_fn = PARAMS['activation_fn']
        normalization_fn = PARAMS['normalization_fn']
        compression = PARAMS['compression']

        # print("in_channels: %s, compression: %s" % (in_channels, compression))
        assert in_channels % compression == 0

        self.in_channels = in_channels
        self.compression = compression

        self.add_module('norm', normalization_fn(in_channels))
        self.add_module('act', activation_fn())
        if not is_last_block:
            self.add_module('conv', nn.Conv2d(in_channels, in_channels // compression, kernel_size=(1, 1),
                                              stride=1, padding=0, bias=True))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        else:
            self.compression = 1

    @property
    def out_channels(self):
        return self.in_channels // self.compression


class Lambda(nn.Module):
    def __init__(self, lambda_fn):
        super(Lambda, self).__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)



class DenseSharp(nn.Module):

    def __init__(self):

        super(DenseSharp, self).__init__()

        input_channels = PARAMS['input_channels']
        input_transform_fn = PARAMS['input_transform_fn']
        input_conv_channels = PARAMS['input_conv_channels']
        normalization_fn = PARAMS['normalization_fn']
        activation_fn = PARAMS['activation_fn']
        down_structure = PARAMS['down_structure']
        output_channels = PARAMS['output_channels']  # type: Any

        self.input_features = nn.Sequential()
        if input_transform_fn is not None:
            self.input_features.add_module("input_transform", Lambda(input_transform_fn))
        self.input_features.add_module("init_conv", nn.Conv2d(input_channels, input_conv_channels, kernel_size=3,
                                                              stride=1, padding=1, bias=True))
        self.input_features.add_module("init_norm", normalization_fn(input_conv_channels))
        self.input_features.add_module("init_act", activation_fn())

        self.dense_blocks = nn.ModuleList()
        self.transmit_blocks = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        channels = input_conv_channels
        for i, num_layers in enumerate(down_structure):
            dense_block = nn.Sequential()
            for j in range(num_layers):
                conv_layer = ConvBlock(channels)##
                dense_block.add_module('denseblock{}_layer{}'.format(i + 1, j + 1), conv_layer)
                channels = conv_layer.out_channels
            self.dense_blocks.append(dense_block)

            if i == 0:
                segmentation_channels = channels
            else:
                # TODO: a pre-activated top-down path
                self.top_down_blocks.append(nn.Sequential(normalization_fn(channels), activation_fn(),
                                                          nn.ConvTranspose2d(channels, up_channels,
                                                                             kernel_size=2, stride=2)))

            trans_layer = TransmitBlock(channels, is_last_block=i == len(down_structure) - 1)
            self.transmit_blocks.append(trans_layer)
            up_channels = channels
            self.channels = trans_layer.out_channels
            channels = self.channels

            #print('channels:',channels)
            #print('output_channels:',output_channels)
            self.classification_head = nn.Linear(channels*32, output_channels)
            self.segmentation_head = nn.Conv2d(segmentation_channels, 1, kernel_size=1, stride=1)

    def forward(self, x, **return_opts):
        batch_size, _, d, h, w = x.size()#32，1，32，32，32
        #transform [N,C,D, H, W] to [N*D,C,H,W]
        #size = x.size()
        x = to_tsm(x,batch_size, _, d, h, w)
        #x = tsm(x,'zero')
        conv = self.input_features(x)#初始化 2d

        # bottom-up for classification
        bottom_up_feats = []
        for dense_block, transmit_block in zip(self.dense_blocks, self.transmit_blocks):
            bottom_up = dense_block(conv)
            #print('bottom_up:',bottom_up.size())
            bottom_up_feats.append(bottom_up)
            conv = transmit_block(bottom_up)###error
            #print('conv:',conv.size())


        # top-down for segmentation
        top_down = bottom_up_feats[-1]
        for bottom_up_feat, top_down_block in zip(reversed(bottom_up_feats[:-1]), reversed(self.top_down_blocks)):
            deconv = top_down_block(top_down)
            top_down = bottom_up_feat + deconv
            #print('top_down:',top_down.size())
        logit = self.segmentation_head(top_down)
        logit = to_seg(logit,d)#(N*D,1,H,W) to (N,D,H,W)
        seg = torch.sigmoid(logit)

        pooled = F.adaptive_avg_pool2d(conv, 1).view(batch_size, -1)
        #print(pooled.size())
        clf = self.classification_head(pooled)##

        ret = dict(clf=clf, seg=seg)

        return ret


if __name__ == '__main__':


