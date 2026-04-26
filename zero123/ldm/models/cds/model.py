import torch
import numpy as np
import torch.nn as nn
import ldm.models.cds.layers as layers
from functools import partial
import torch.nn.functional as F

#np.random.seed(0)
#torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)


def count_params(model): return sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)


def convert_mag_cos_sin(inputs, eps=1e-9):
    # Converts a [B, C, H, W] input into a [B, 3, C, H, W] complex representation
    mag = torch.abs(inputs)
    phase = torch.angle(inputs)
    return torch.stack([torch.log(mag+eps), torch.cos(phase), torch.sin(phase)], dim=1)


class complex2real(nn.Module):
    def __init__(self):
        super(complex2real, self).__init__()

    def forward(self, x):
        return torch.stack([x.real, x.imag], dim=1)


class real2complex(nn.Module):
    def __init__(self):
        super(real2complex, self).__init__()

    def forward(self, x):
        return x[:, 0]+1j*x[:, 1]


class twentyk_cnn(nn.Module):
    # Backbone 20K cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(twentyk_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 30, (3, 3), stride=(2,2), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 30, (3, 3), stride=(2,2), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 30, (3, 3), stride=(2,2), groups=groups)

        self.conv_2 = nn.Conv2d(40, 50, (3, 3), (2, 2), groups=groups)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.conv_3 = nn.Conv2d(50, 60, (3, 3), (2,2), groups=groups)
        self.bn_3 = nn.GroupNorm(12, 60)
        self.linear_2 = nn.Linear(60, 30)
        out_size = 10 if no_clutter else 11
        self.linear_4 = nn.Linear(30, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 50))
        self.id2 = nn.Conv2d(50, 50, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 5), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(2, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         int(out_channel / 5), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(2, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class tenk_cnn(nn.Module):
    # Backbone 10k cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(tenk_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 20, (3, 3), stride=(2,2), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 20, (3, 3), stride=(2,2), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 20, (3, 3), stride=(2,2), groups=groups)

        self.conv_2 = nn.Conv2d(30, 40, (3, 3), (2, 2), groups=groups*2)
        self.bn_1 = nn.GroupNorm(5, 20)
        self.bn_2 = nn.GroupNorm(10, 40)
        self.linear_2 = nn.Linear(50, 50)
        out_size = 10 if no_clutter else 11
        self.linear_4 = nn.Linear(50, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(20, 30))
        self.id1 = nn.Conv2d(20, 30, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(40, 50))
        self.id2 = nn.Conv2d(40, 50, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 5), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(2, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         int(out_channel / 5), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(2, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class fivek_cnn(nn.Module):
    # Backbone 5k cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(fivek_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 20, (3, 3), stride=(2,2), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 20, (3, 3), stride=(2,2), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 20, (3, 3), stride=(2,2), groups=groups)

        self.conv_2 = nn.Conv2d(25, 25, (3, 3), (2, 2), groups=groups)
        self.bn_1 = nn.GroupNorm(5, 20)
        self.bn_2 = nn.GroupNorm(5, 25)
        self.linear_2 = nn.Linear(30, 20)
        out_size = 10 if no_clutter else 11
        self.linear_4 = nn.Linear(20, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(20, 25))
        self.id1 = nn.Conv2d(20, 25, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(25, 30))
        self.id2 = nn.Conv2d(25, 30, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 5), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(1, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         int(out_channel / 5), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(1, int(out_channel / 5)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 5),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class two00_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, groups=3, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(two00_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 3, (1, 1), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 3, (1, 1), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 9, (1, 1), groups=groups)

        self.bn_1 = nn.GroupNorm(3, 9)
        out_size = 10 if no_clutter else 11
        self.linear_2 = nn.Linear(9, out_size)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        return x


class six00_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, groups=3, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(six00_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 12, (3, 3), stirde=(2,2), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 12, (3, 3), stride=(2,2), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 12, (3, 3), stride=(2,2), groups=groups)

        self.bn_1 = nn.GroupNorm(3, 12)
        out_size = 10 if no_clutter else 11
        self.linear_2 = nn.Linear(12, out_size)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        return x


class tiny_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(tiny_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 15, (3, 3), stirde=(2,2), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 15, (3, 3), stride=(2,2), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 15, (3, 3), stride=(2,2), groups=groups)

        self.conv_2 = nn.Conv2d(15, 15, (3, 3), stride=(2, 2), groups=groups)
        self.bn_1 = nn.GroupNorm(5, 15)
        self.bn_2 = nn.GroupNorm(5, 15)
        out_size = 10 if no_clutter else 11
        self.linear_2 = nn.Linear(15, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(15, 15))
        self.id1 = nn.Conv2d(15, 15, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 3), (1, 1), bias=False, groups=5))
        res_block.append(nn.GroupNorm(1, int(out_channel / 3)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 3),
                         int(out_channel), (3, 3), bias=False, padding=1, groups=5))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        return x


class small_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(small_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 30, (5, 5), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 30, (5, 5), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 30, (5, 5), groups=groups)

        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3), groups=groups)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), groups=groups)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.linear_2 = nn.Linear(70, 30)
        out_size = 10 if no_clutter else 11
        self.linear_4 = nn.Linear(30, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class FRUITS_RGB(nn.Module):
    """
    FRUITS Model (RGB) for small FRUITS experiments.
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, dset_type="rgb", outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_RGB, self).__init__()

        # Building layers....
        conv = nn.Conv2d
        self.dset_type = dset_type

        assert(dset_type in ["rgb","gray"])
        inp_size = 3 if dset_type == "rgb" else 1
        self.wfm1 = conv(inp_size, 4, kernel_size=7, stride=(4, 4))
        self.wfm2 = conv(4, 6, kernel_size=5, stride=(3, 3), groups=1)
        self.wfm3 = conv(6, 8, kernel_size=3, stride=(2, 2), groups=1)
        self.wfm4 = conv(8, 16, kernel_size=3, stride=(2,2), groups=2)
        self.prelu1 = nn.PReLU(4)
        self.prelu2 = nn.PReLU(6)
        self.prelu3 = nn.PReLU(8)

        self.fc1 = conv(16, prototype_size*2, 1, groups=1)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        if self.dset_type == 'gray':
          x = torch.mean(x, dim=1, keepdim=True)
        x = self.wfm1(x) #[]
        x = self.prelu1(x)
        x = self.wfm2(x) #[]
        x = self.prelu2(x)
        x = self.wfm3(x) #[
        x = self.prelu3(x)
        x = self.wfm4(x) #[

        x = self.fc1(x) #[
        x_shape = x.shape
        #from IPython import embed;embed()
        if x_shape[0] > 1:
          x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape[0],2,x_shape[1]//2) # [B,2,P]
        else: # works for batch size 1
          x = x.reshape(x_shape[0], 2, x_shape[1]//2)
        x = self.dist_feat(x)

        return x


class FRUITS_RGB_BIG(nn.Module):
    """
    FRUITS Model (RGB) for small FRUITS experiments.
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, dset_type="rgb", outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_RGB_BIG, self).__init__()

        # Building layers....
        conv = nn.Conv2d
        self.dset_type = dset_type

        assert(dset_type in ["rgb","gray"])
        inp_size = 3 if dset_type == "rgb" else 1
        self.wfm1 = conv(inp_size, 16, kernel_size=3, stride=(2, 2))
        self.wfm2 = conv(16, 32, kernel_size=5, stride=(3, 3), groups=4)
        self.wfm3 = conv(32, 32, kernel_size=3, stride=(2, 2), groups=16)
        self.wfm4 = conv(32, 64, kernel_size=3, stride=(2,2), groups=16)
        self.prelu1 = nn.PReLU(16)
        self.prelu2 = nn.PReLU(32)
        self.prelu3 = nn.PReLU(32)

        self.fc1 = conv(64, prototype_size*2, 4, groups=64)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        if self.dset_type == 'gray':
          x = torch.mean(x, dim=1, keepdim=True)
        x = self.wfm1(x) #[B,16,63,63]
        x = self.prelu1(x)
        x = self.wfm2(x) #[B,32,20,20]
        x = self.prelu2(x)
        x = self.wfm3(x) #[B,32,9,9]
        x = self.prelu3(x)
        x = self.wfm4(x) #[B,64,4,4]

        x = self.fc1(x) #[B,P*2,1,1]
        x_shape = x.shape
        x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape[0],2,x_shape[1]//2) # [B,2,P]
        x = self.dist_feat(x)

        return x

class FRUITS_I(nn.Module):
    """
    FRUITS Model (I-Type) for small FRUITS experiments. Based on CIFARnet
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_I, self).__init__()

        # Building layers....
        conv = layers.ComplexConv
        diff = layers.DivLayer

        inp_size = 2 if ((dset_type == 'lab') or (
            dset_type == 'sliding')) else 3
        self.wfm1 = conv(inp_size, 4, kernel_size=7, stride=(
            4, 4), new_init=True, use_groups_init=True, bias=False)

        self.wfm2 = conv(4, 6, kernel_size=5,
                         stride=(3, 3), groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(6, 8, kernel_size=3,
                         stride=(2, 2), groups=2, new_init=True, use_groups_init=True)
        self.wfm4 = conv(8, 16, kernel_size=3, stride=(2,2), groups=4,
                         new_init=True, use_groups_init=True)

        self.diff1 = diff(4, 3, new_init=True)

        self.gtrelu1 = layers.GTReLU(4, phase_scale=True)
        self.gtrelu2 = layers.GTReLU(6, phase_scale=True)
        self.gtrelu3 = layers.GTReLU(8, phase_scale=True)

        self.fc1 = conv(16, prototype_size, 1, groups=1, new_init=True)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x) # B,2,4,31,31
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.wfm2(x) # B,2,6,9,9
        x = self.gtrelu2(x)
        x = self.wfm3(x) #B,2,8,4,4
        x = self.gtrelu3(x)
        x = self.wfm4(x) #B,2,16,1,1

        x = self.fc1(x) #B, 2, P, 1,1
        x_shape = x.shape
        x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)
        x = self.dist_feat(x[..., 0, 0])

        return x


class FRUITS_I_BIG(nn.Module):
    """
    FRUITS Model (I-Type) for small FRUITS experiments. Based on CIFARnet
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_I_BIG, self).__init__()

        # Building layers....
        conv = layers.ComplexConv
        diff = layers.DivLayer

        inp_size = 2 if ((dset_type == 'lab') or (
            dset_type == 'sliding')) else 3
        self.wfm1 = conv(inp_size, 16, kernel_size=3, stride=(
            2, 2), new_init=True, use_groups_init=True, bias=False)

        self.wfm2 = conv(16, 32, kernel_size=5,
                         stride=(3, 3), groups=8, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 32, kernel_size=3,
                         stride=(2, 2), groups=32, new_init=True, use_groups_init=True)
        self.wfm4 = conv(32, 64, kernel_size=3, stride=(2,2), groups=32,
                         new_init=True, use_groups_init=True)

        self.diff1 = diff(16, 3, new_init=True)

        self.gtrelu1 = layers.GTReLU(16, phase_scale=True)
        self.gtrelu2 = layers.GTReLU(32, phase_scale=True)
        self.gtrelu3 = layers.GTReLU(32, phase_scale=True)

        self.fc1 = conv(64, prototype_size, 4, groups=64, new_init=True)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)
        x = self.wfm3(x)
        x = self.gtrelu3(x)
        x = self.wfm4(x)

        x = self.fc1(x)
        x_shape = x.shape
        x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)
        x = self.dist_feat(x[..., 0, 0])

        return x


class CDS_I(nn.Module):
    """
    CDS Model (I-Type) for small CIFAR experiments. Based on CIFARnet
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, cifarnet_config='dgtf', dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(CDS_I, self).__init__()
        print("CIFARnet Config:", cifarnet_config)
        self.cifarnet_config = cifarnet_config

        # Building layers....
        conv = layers.ComplexConv
        diff = layers.DivLayer

        inp_size = 2 if ((dset_type == 'lab') or (
            dset_type == 'sliding')) else 3
        self.wfm1 = conv(inp_size, 16, kernel_size=3, stride=(
            2, 2), reflect=1, new_init=True, use_groups_init=True, bias=False)

        self.wfm2 = conv(16, 32, kernel_size=3,
                         stride=(2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 64, kernel_size=3,
                         stride=(2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
        self.wfm4 = conv(64, 64, kernel_size=4, groups=64,
                         new_init=True, use_groups_init=True)

        self.diff1 = diff(16, 3, reflect=1, new_init=True)

        self.gtrelu1 = layers.GTReLU(16, phase_scale=True)
        self.gtrelu2 = layers.GTReLU(32, phase_scale=True)
        self.gtrelu3 = layers.GTReLU(64, phase_scale=True)

        self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)
        x = self.wfm3(x)
        x = self.gtrelu3(x)
        x = self.wfm4(x)

        x = self.fc1(x)
        x_shape = x.shape
        x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)
        x = self.dist_feat(x[..., 0, 0])

        return x


class FRUITS_E(nn.Module):
    """
    FRUITS Model (E-Type) for small FRUITS experiments. Based on CIFARnet
    """

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_E, self).__init__()

        conv = layers.ComplexConv
        inp_size = 2 if (dset_type in ['lab','sliding']) else 3
        self.wfm1 = conv(inp_size, 4, kernel_size=7, stride=(
            4, 4), new_init=True, use_groups_init=True)
        self.wfm2 = conv(4, 6, kernel_size=5, stride=(
            3, 3), groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(6, 8, kernel_size=3, stride=(
            2, 2), groups=2, new_init=True, use_groups_init=True)
        self.wfm4 = conv(8, 16, kernel_size=3, stride=(2,2), groups=4,
                         new_init=True, use_groups_init=True)

        self.s1 = layers.scaling_layer(4)
        self.s2 = layers.scaling_layer(6)
        self.s3 = layers.scaling_layer(8)

        self.t1 = layers.eqnl(
            4, clampdiv=True, groups=1)
        self.t2 = layers.eqnl(
            6, clampdiv=True, groups=1)
        self.t3 = layers.eqnl(
            8, clampdiv=True, groups=1)

        self.fc1 = conv(16, prototype_size, 1, groups=1, new_init=True)

        self.dist_feat = layers.DistFeatures(prototype_size, outsize)

        self.bn = layers.VNCBN(prototype_size)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        # [B, 2,128,128], complex -> [B,2,2,128,128] float
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x) #
        x = self.s1(x)
        x = self.t1(x)
        x = self.wfm2(x) #
        x = self.s2(x)
        x = self.t2(x)
        x = self.wfm3(x) #
        x = self.s3(x)
        x = self.t3(x)
        x = self.wfm4(x) #

        x = self.fc1(x) #[

        x = self.bn(x)

        y = torch.sum(x, dim=2, keepdim=True)/np.sqrt(x.shape[2]*2) #TODO??
        x = self.dist_feat(x[..., 0, 0], y)

        return x


class FRUITS_E_BIG(nn.Module):
    """
    FRUITS Model (E-Type) for small FRUITS experiments. Based on CIFARnet
    """

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(FRUITS_E_BIG, self).__init__()

        conv = layers.ComplexConv
        inp_size = 2 if (dset_type in ['lab','sliding']) else 3
        self.wfm1 = conv(inp_size, 16, kernel_size=3, stride=(
            2, 2), new_init=True, use_groups_init=True)
        self.wfm2 = conv(16, 32, kernel_size=5, stride=(
            3, 3), groups=8, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 32, kernel_size=3, stride=(
            2, 2), groups=32, new_init=True, use_groups_init=True)
        self.wfm4 = conv(32, 64, kernel_size=3, stride=(2,2), groups=32,
                         new_init=True, use_groups_init=True)

        self.s1 = layers.scaling_layer(16)
        self.s2 = layers.scaling_layer(32)
        self.s3 = layers.scaling_layer(32)

        self.t1 = layers.eqnl(
            16, clampdiv=True, groups=1)
        self.t2 = layers.eqnl(
            32, clampdiv=True, groups=1)
        self.t3 = layers.eqnl(
            32, clampdiv=True, groups=1)

        self.fc1 = conv(64, prototype_size, 4, groups=64, new_init=True)

        self.dist_feat = layers.DistFeatures(prototype_size, outsize)

        self.bn = layers.VNCBN(prototype_size)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        # [B, 2,128,128], complex -> [B,2,2,128,128] float
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x) #[B,2,16,63,63]
        x = self.s1(x)
        x = self.t1(x)
        x = self.wfm2(x) #[B,2,32,20,20]
        x = self.s2(x)
        x = self.t2(x)
        x = self.wfm3(x) #[B,2,64,9,9]
        x = self.s3(x)
        x = self.t3(x)
        x = self.wfm4(x) #[B,2,64,4,4]

        x = self.fc1(x) #[B,2,128,1,1]

        x = self.bn(x)

        y = torch.sum(x, dim=2, keepdim=True)/np.sqrt(x.shape[2]*2) #TODO??
        x = self.dist_feat(x[..., 0, 0], y)

        return x


class CDS_E(nn.Module):
    """
    CDS Model (E-Type) for small CIFAR experiments. Based on CIFARnet
    """

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(CDS_E, self).__init__()

        conv = layers.ComplexConv
        inp_size = 2 if (dset_type == 'lab') else 3
        self.wfm1 = conv(inp_size, 16, kernel_size=3, stride=(
            2, 2), reflect=1, new_init=True, use_groups_init=True)
        self.wfm2 = conv(16, 32, kernel_size=3, stride=(
            2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 64, kernel_size=3, stride=(
            2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
        self.wfm4 = conv(64, 64, kernel_size=4, groups=64,
                         new_init=True, use_groups_init=True)

        self.s1 = layers.scaling_layer(16)
        self.s2 = layers.scaling_layer(32)
        self.s3 = layers.scaling_layer(64)

        self.t1 = layers.eqnl(
            16, clampdiv=True, groups=1)
        self.t2 = layers.eqnl(
            32, clampdiv=True, groups=1)
        self.t3 = layers.eqnl(
            64, clampdiv=True, groups=1)

        self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)

        df_conv = conv(prototype_size, prototype_size, kernel_size=1, groups=16,
                       new_init=True, use_groups_init=True)

        self.dist_feat = layers.DistFeatures(prototype_size, outsize)

        self.bn = layers.VNCBN(prototype_size)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.s1(x)
        x = self.t1(x)
        x = self.wfm2(x)
        x = self.s2(x)
        x = self.t2(x)
        x = self.wfm3(x)
        x = self.s3(x)
        x = self.t3(x)
        x = self.wfm4(x)

        x = self.fc1(x)

        x = self.bn(x)

        y = torch.sum(x, dim=2, keepdim=True)/np.sqrt(x.shape[2]*2)
        x = self.dist_feat(x[..., 0, 0], y)

        return x

class CDS_200_MSTAR(nn.Module):
    """
    MSTAR 200 Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
        super(CDS_200_MSTAR, self).__init__()
        # for groups in cnn backbone
        groups = 3

        conv = layers.ComplexConv
        diff = layers.ConjugateLayer
        self.wfm1 = conv(1, 1, 5, (5, 5), groups=1)
        self.diff1 = diff(1, 1, groups=1)
        self.gtrelu1 = layers.GTReLU(1)
        self.mp = layers.MaxPoolMag(2)
        self.cnn = two00_cnn(groups=groups, no_clutter=no_clutter, in_size=3)

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)

        mag = torch.norm(x, dim=1)
        phase = torch.atan2(x[:, 1, ...], x[:, 0, ...])

        mag = mag + 1e-5
        log_mag = torch.log(mag)

        log_mag = log_mag.unsqueeze(1)

        cos = torch.cos(phase)
        cos = cos.unsqueeze(1)

        sin = torch.sin(phase)
        sin = sin.unsqueeze(1)
        x = torch.cat([log_mag, cos, sin], dim=1)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],
                      x.shape[-2], x.shape[-1])
        return self.cnn(x)


class CDS_600_MSTAR(nn.Module):
    """
    MSTAR 200 Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
        super(CDS_600_MSTAR, self).__init__()
        # for groups in cnn backbone
        groups = 3

        conv = layers.ComplexConv
        diff = layers.ConjugateLayer
        self.wfm1 = conv(1, 3, 3, (3, 3), groups=1)
        self.diff1 = diff(3, 1, groups=1)
        self.gtrelu1 = layers.GTReLU(3)
        self.mp = layers.MaxPoolMag(2)
        self.wfm2 = conv(3, 3, 3, (3, 3), groups=3)
        self.gtrelu2 = layers.GTReLU(3)
        self.cnn = six00_cnn(groups=groups, no_clutter=no_clutter, in_size=9)

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)

        mag = torch.norm(x, dim=1)
        phase = torch.atan2(x[:, 1, ...], x[:, 0, ...])

        mag = mag + 1e-5
        log_mag = torch.log(mag)

        log_mag = log_mag.unsqueeze(1)

        cos = torch.cos(phase)
        cos = cos.unsqueeze(1)

        sin = torch.sin(phase)
        sin = sin.unsqueeze(1)
        x = torch.cat([log_mag, cos, sin], dim=1)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],
                      x.shape[-2], x.shape[-1])
        return self.cnn(x)


class CDS_TINY_MSTAR(nn.Module):
    """
    MSTAR TINY Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
        super(CDS_TINY_MSTAR, self).__init__()
        # for groups in cnn backbone
        groups = 5

        conv = layers.ComplexConv
        diff = layers.ConjugateLayer
        self.wfm1 = conv(1, 5, 3, (2, 2), groups=1)
        self.diff1 = diff(5, 1, groups=1)
        self.gtrelu1 = layers.GTReLU(5)
        self.mp = layers.MaxPoolMag(2)
        self.wfm2 = conv(5, 5, 3, (2, 2), groups=1)
        self.gtrelu2 = layers.GTReLU(5)
        self.cnn = tiny_cnn(groups=groups, no_clutter=no_clutter, in_size=15)

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)

        mag = torch.norm(x, dim=1)
        phase = torch.atan2(x[:, 1, ...], x[:, 0, ...])

        mag = mag + 1e-5
        log_mag = torch.log(mag)

        log_mag = log_mag.unsqueeze(1)

        cos = torch.cos(phase)
        cos = cos.unsqueeze(1)

        sin = torch.sin(phase)
        sin = sin.unsqueeze(1)
        x = torch.cat([log_mag, cos, sin], dim=1)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],
                      x.shape[-2], x.shape[-1])
        return self.cnn(x)


class CDS_10K_MSTAR(CDS_TINY_MSTAR):
    """
    MSTAR 10K Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
      super(CDS_10K_MSTAR, self).__init__(no_clutter=no_clutter, *args, **kwargs)
      groups = 5
      self.cnn = tenk_cnn(groups=groups, no_clutter=no_clutter, in_size=15)


class CDS_5K_MSTAR(CDS_TINY_MSTAR):
    """
    MSTAR 10K Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
      super(CDS_5K_MSTAR, self).__init__(no_clutter=no_clutter, *args, **kwargs)
      groups = 5
      self.cnn = fivek_cnn(groups=groups, no_clutter=no_clutter, in_size=15)


class CDS_MSTAR(nn.Module):
    """
    MSTAR Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
        super(CDS_MSTAR, self).__init__()
        # for groups in cnn backbone
        groups = 5

        conv = layers.ComplexConv
        diff = layers.ConjugateLayer
        self.wfm1 = conv(1, 5, 5, (1, 1), groups=1)
        self.diff1 = diff(5, 3, groups=1)
        self.gtrelu1 = layers.GTReLU(5)
        self.mp = layers.MaxPoolMag(2)
        self.wfm2 = conv(5, 5, 3, (2, 2), groups=1)
        self.gtrelu2 = layers.GTReLU(5)
        self.cnn = small_cnn(groups=groups, no_clutter=no_clutter, in_size=15)

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)

        mag = torch.norm(x, dim=1)
        phase = torch.atan2(x[:, 1, ...], x[:, 0, ...])

        mag = mag + 1e-5
        log_mag = torch.log(mag)

        log_mag = log_mag.unsqueeze(1)

        cos = torch.cos(phase)
        cos = cos.unsqueeze(1)

        sin = torch.sin(phase)
        sin = sin.unsqueeze(1)
        x = torch.cat([log_mag, cos, sin], dim=1)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],
                      x.shape[-2], x.shape[-1])
        return self.cnn(x)


class CDS_20K_MSTAR(CDS_MSTAR):
    """
    MSTAR 10K Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
      super(CDS_20K_MSTAR, self).__init__(no_clutter=no_clutter, *args, **kwargs)
      groups = 5
      self.cnn = twentyk_cnn(groups=groups, no_clutter=no_clutter, in_size=15)


def conv_bn_complex(c_in, c_out, groups=1):
    return nn.Sequential(
        layers.ComplexConvFast(c_in, c_out, kernel_size=3,
                               padding=1, groups=groups),
        layers.ComplexBN(c_out),
        nn.ReLU(True),
    )


class residual_complex(nn.Module):
    def __init__(self, c, groups=1):
        super(residual_complex, self).__init__()
        self.res = nn.Sequential(
            conv_bn_complex(c, c, groups=groups),
            conv_bn_complex(c, c, groups=groups)
        )

    def forward(self, x):
        return x + self.res(x)


class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class mul(nn.Module):
    def __init__(self, c):
        super(mul, self).__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


def CDS_large(outsize=10, *args, **kwargs):
    channels = {'prep': 64,
                'layer1': 128, 'layer2': 256, 'layer3': 256}
    n = [
        layers.ComplexConvFast(2, channels['prep'], kernel_size=3, padding=1, groups=1),

        layers.ConjugateLayer(channels['prep'], kernel_size=1, use_one_filter=True),

        conv_bn_complex(channels['prep'], channels['prep'], groups=2),
        conv_bn_complex(channels['prep'], channels['layer1'], groups=2),
        layers.MaxPoolMag(2),
        residual_complex(channels['layer1'], groups=2),
        conv_bn_complex(channels['layer1'], channels['layer2'], groups=4),
        layers.MaxPoolMag(2),
        conv_bn_complex(channels['layer2'], channels['layer3'], groups=2),
        layers.MaxPoolMag(2),
        residual_complex(channels['layer3'], groups=4),
        layers.MaxPoolMag(4),
        flatten(),
        nn.Linear(channels['layer3']*2, outsize, bias=False),
        mul(0.125),
    ]
    return nn.Sequential(*n)


class ComplexBasicBlock(nn.Module):

  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1, groups=1, i_index=None, is_complex=True):
    super().__init__()

    conv = layers.ComplexConvFast if is_complex else nn.Conv2d
    diff = layers.ConjugateLayer if is_complex else nn.Identity
    bn = layers.ComplexBN if is_complex else nn.BatchNorm2d
    self.i_index = i_index
    self.groups = groups

    residual_function = [
        conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups),
        bn(out_channels),
        nn.ReLU(inplace=True),
        conv(out_channels, out_channels * ComplexBasicBlock.expansion, kernel_size=3, padding=1, bias=False, groups=groups),
        bn(out_channels*ComplexBasicBlock.expansion)
    ]

    if i_index is not None:
      assert(i_index in [1,4]), f"i_index: {i_index} is not allowed! Only in [1,4]."
      residual_function.insert(i_index, diff(out_channels, kernel_size=1, use_one_filter=True, groups=groups))
      self.shortcut = nn.Sequential(diff(in_channels, kernel_size=1, use_one_filter=True, groups=groups))
    else:
      self.shortcut = nn.Sequential()

    self.residual_function = nn.Sequential(*residual_function)

    if stride != 1 or in_channels != ComplexBasicBlock.expansion * out_channels:
      if i_index is None:
        self.shortcut = nn.Sequential(
           conv(in_channels, out_channels*ComplexBasicBlock.expansion, kernel_size=1, stride=stride, bias=False, groups=groups),
           bn(out_channels * ComplexBasicBlock.expansion),
        )
      else:
        self.shortcut = nn.Sequential(
           conv(in_channels, out_channels*ComplexBasicBlock.expansion, kernel_size=1, stride=stride, bias=False, groups=groups),
           diff(out_channels * ComplexBasicBlock.expansion, kernel_size=1, use_one_filter=True, groups=groups),
           bn(out_channels * ComplexBasicBlock.expansion)
        )


  def forward(self, x):
    return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ComplexBottleNeck(nn.Module):

  expansion = 4

  def __init__(self, in_channels, out_channels, stride=1, groups=1, i_index=None, is_complex=True):
    super().__init__()

    conv = layers.ComplexConvFast if is_complex else nn.Conv2d
    diff = layers.ConjugateLayer if is_complex else nn.Identity
    bn = layers.ComplexBN if is_complex else nn.BatchNorm2d
    self.i_index = i_index
    self.groups = groups

    residual_function = [
        conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups),
        bn(out_channels),
        nn.ReLU(inplace=True),
        conv(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False, groups=groups),
        bn(out_channels),
        nn.ReLU(inplace=True),
        conv(out_channels, out_channels * ComplexBottleNeck.expansion, kernel_size=1, bias=False, groups=groups),
        bn(out_channels * ComplexBottleNeck.expansion),
    ]

    if i_index is not None:
      assert(i_index in [1,4,7]), f"i_index: {i_index} is not allowed! Only in [1,4,7]."
      C_out = out_channels
      if i_index == 7:
        C_out = out_channels * ComplexBottleNeck.expansion
      residual_function.insert(i_index, diff(C_out, kernel_size=1, use_one_filter=True, groups=groups))
      self.shortcut = nn.Sequential(diff(in_channels, kernel_size=1, use_one_filter=True, groups=groups))
    else:
      self.shortcut = nn.Sequential()

    self.residual_function = nn.Sequential(*residual_function)

    if stride != 1 or in_channels != out_channels * ComplexBottleNeck.expansion:
      if i_index is None:
        self.shortcut = nn.Sequential(
            conv(in_channels, out_channels * ComplexBottleNeck.expansion, stride=stride, kernel_size=1, bias=False, groups=groups),
            bn(out_channels * ComplexBottleNeck.expansion)
        )
      else:
        self.shortcut = nn.Sequential(
            conv(in_channels, out_channels * ComplexBottleNeck.expansion, stride=stride, kernel_size=1, bias=False, groups=groups),
            diff(out_channels * ComplexBottleNeck.expansion, kernel_size=1, use_one_filter=True, groups=groups),
            bn(out_channels * ComplexBottleNeck.expansion)
        )

  def forward(self, x):
    return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ComplexResNet(nn.Module):
  def __init__(self, block, num_block, num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=1.0, is_complex=True, inp_channels=None):
    super().__init__()

    conv = layers.ComplexConvFast if is_complex else nn.Conv2d
    diff = layers.ConjugateLayer if is_complex else nn.Identity #throws an error
    bn = layers.ComplexBN if is_complex else nn.BatchNorm2d
    self.i_layer = i_layer
    self.i_block = i_block
    self.i_index = i_index
    self.groups = groups
    self.is_complex = is_complex

    self.in_channels = int(64/divisor)
    if inp_channels is None:
      inp_channels = 2 if is_complex else 3

    conv1 = [
        conv(inp_channels, int(64/divisor), kernel_size=3, padding=1, bias=False, groups=groups),
        bn(int(64/divisor)),
        nn.ReLU(inplace=True)
    ]

    if i_layer is None and i_block is None and i_index == 1:
      print("Invariance in first conv")
      conv1.insert(1, diff(int(64/divisor), kernel_size=1, use_one_filter=True, groups=groups))

    self.conv1 = nn.Sequential(*conv1)

    self.conv2_x = self._make_layer(block, int(64/divisor), num_block[0], 1, 0, i_block, i_index)
    self.conv3_x = self._make_layer(block, int(128/divisor), num_block[1], 2, 1, i_block, i_index)
    self.conv4_x = self._make_layer(block, int(256/divisor), num_block[2], 2, 2, i_block, i_index)
    self.conv5_x = self._make_layer(block, int(512/divisor), num_block[3], 2, 3, i_block, i_index)

    self.avg_pool = layers.ComplexAdaptiveAvgPool2d() if is_complex else nn.AdaptiveAvgPool2d((1,1))
    if is_complex:
      self.fc = nn.Linear(int(512/divisor)*2*block.expansion, num_classes)
    else:
      self.fc = nn.Linear(int(512/divisor)*block.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride, i_layer, i_block, i_index):

    strides = [stride] + [1] * (num_blocks -1)
    layers = []
    for i, stride in enumerate(strides):
      _i_index = None
      if i_layer == self.i_layer and i_block == i and i_index is not None:
        print(f"Invariance in  layer: {i_layer}, block: {i}, index: {i_index}")
        _i_index = i_index
      layers.append(block(self.in_channels, out_channels, stride=stride, groups=self.groups, i_index=_i_index, is_complex=self.is_complex))
      self.in_channels = out_channels * block.expansion

    return nn.Sequential(*layers)

  def forward(self, x):
    output = self.conv1(x)
    output = self.conv2_x(output)
    output = self.conv3_x(output)
    output = self.conv4_x(output)
    output = self.conv5_x(output)
    output = self.avg_pool(output)
    output = output.view(output.size(0), -1)
    output = self.fc(output)

    return output


def complex_resnet18(num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=1.41, is_complex=True, inp_channels=None): #11.28M
  return ComplexResNet(ComplexBasicBlock, [2,2,2,2], num_classes, groups, i_layer, i_block, i_index, divisor, is_complex, inp_channels=inp_channels)

def complex_resnet34(num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=1.4, is_complex=True, inp_channels=None): #21.6M
  return ComplexResNet(ComplexBasicBlock, [3,4,6,3], num_classes, groups, i_layer, i_block, i_index, divisor, is_complex, inp_channels=inp_channels)

def complex_resnet50(num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=2.22, is_complex=True, inp_channels=None): #23.65M
  return ComplexResNet(ComplexBottleNeck, [3,4,6,3], num_classes, groups, i_layer, i_block, i_index, divisor, is_complex, inp_channels=inp_channels)

def complex_resnet101(num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=2.3, is_complex=True, inp_channels=None): #42.61M
  return ComplexResNet(ComplexBottleNeck, [3,4,23,3], num_classes, groups, i_layer, i_block, i_index, divisor, is_complex, inp_channels=inp_channels)

def complex_resnet152(num_classes=100, groups=1, i_layer=None, i_block=None, i_index=None, divisor=2.32, is_complex=True, inp_channels=None): #58.54M
  return ComplexResNet(ComplexBottleNeck, [3,8,36,3], num_classes, groups, i_layer, i_block, i_index, divisor, is_complex, inp_channels=inp_channels)


class Classifier(nn.Module):
  def __init__(self, num_classes=10, divisor=1., is_complex=False, fix_backbone=True, is_regression=False):
    super().__init__()

    self.backbone = complex_resnet18(num_classes=num_classes, divisor=divisor, is_complex=is_complex)
    #self.backbone.fc = torch.nn.Identity()

    if fix_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False
      self.fix_backbone_batchnorms()

    #hidden_dim = 512
    #if is_complex:
    #  hidden_dim = 726 #int(512/1.41)*2 TODO: fixed

    #if is_regression:
    #  self.fc = nn.Linear(hidden_dim, 1) # only azimuth for now
      #self.fc = nn.Sequential(
      #    nn.Linear(hidden_dim, 128),
      #    nn.ReLU(),
      #    nn.Linear(128,1)
      #)
      #self.criterion = nn.MSELoss()
    #else:
    #  self.fc = nn.Linear(hidden_dim, num_classes)
      #self.criterion = nn.CrossEntropyLoss()
    #self.validation_step_outputs = []

  def fix_backbone_batchnorms(self):
    for module in self.backbone.modules():
      if isinstance(module, torch.nn.modules.BatchNorm1d):
        module.eval()
      if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval()
      if isinstance(module, torch.nn.modules.BatchNorm3d):
        module.eval()

  def forward(self, x):
    y_hat = self.backbone(x).flatten(start_dim=1)
    #y_hat = self.fc(y_hat)
    return y_hat

