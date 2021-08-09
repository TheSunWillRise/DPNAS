import torch
import torch.nn as nn
import torch.nn.functional as F
import math


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),

    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),

    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),

    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),

    'resep_conv_3x3': lambda C, stride, affine: ResSepConv(C, C, 3, stride, 1, affine=affine),
    'resep_conv_5x5': lambda C, stride, affine: ResSepConv(C, C, 5, stride, 2, affine=affine),



    ######################## For DP ########################
    'priv_avg_pool_3x3': lambda C, stride, affine:
         nn.AvgPool2d(3, stride=stride, padding=1),
    'priv_max_pool_3x3': lambda C, stride, affine:
         nn.MaxPool2d(3, stride=stride, padding=1),

    'priv_skip_connect': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, None),
    'priv_skip_connect_relu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, relu()),
    'priv_skip_connect_elu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, elu()),
    'priv_skip_connect_tanh': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, tanh()),
    'priv_skip_connect_selu': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, selu()),
    'priv_skip_connect_sigmoid': lambda C, stride, affine:
        Identity() if stride == 1 else PrivFactorizedReduce(C, C, sigmoid()),

    'priv_dil_conv_3x3_relu': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, relu()),
    'priv_dil_conv_3x3_elu': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, elu()),
    'priv_dil_conv_3x3_tanh': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, tanh()),
    'priv_dil_conv_3x3_sigmoid': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, sigmoid()),
    'priv_dil_conv_3x3_selu': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, selu()),
    'priv_dil_conv_3x3_htanh': lambda C, stride, affine:
    PrivDilConv(C, C, 3, stride, 2, 2, htanh()),

    'priv_sep_conv_3x3_relu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, relu()),
    'priv_sep_conv_3x3_elu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, elu()),
    'priv_sep_conv_3x3_tanh': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, tanh()),
    'priv_sep_conv_3x3_sigmoid': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, sigmoid()),
    'priv_sep_conv_3x3_selu': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, selu()),
    'priv_sep_conv_3x3_htanh': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, htanh()),
    'priv_sep_conv_3x3_linear': lambda C, stride, affine:
        PrivSepConv(C, C, 3, stride, 1, Identity()),

    'priv_resep_conv_3x3_relu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, relu()),
    'priv_resep_conv_3x3_elu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, elu()),
    'priv_resep_conv_3x3_tanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, tanh()),
    'priv_resep_conv_3x3_sigmoid': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, sigmoid()),
    'priv_resep_conv_3x3_selu': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, selu()),
    'priv_resep_conv_3x3_htanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, htanh()),
    'priv_resep_conv_3x3_linear': lambda C, stride, affine:
        PrivResSepConv(C, C, 3, stride, 1, Identity()),

    'priv_sep_conv_5x5_relu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, relu()),
    'priv_sep_conv_5x5_elu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, elu()),
    'priv_sep_conv_5x5_tanh': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, tanh()),
    'priv_sep_conv_5x5_sigmoid': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, sigmoid()),
    'priv_sep_conv_5x5_selu': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, selu()),
    'priv_sep_conv_5x5_htanh': lambda C, stride, affine:
        PrivSepConv(C, C, 5, stride, 2, htanh()),

    'priv_resep_conv_5x5_relu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, relu()),
    'priv_resep_conv_5x5_elu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, elu()),
    'priv_resep_conv_5x5_tanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, tanh()),
    'priv_resep_conv_5x5_sigmoid': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, sigmoid()),
    'priv_resep_conv_5x5_selu': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, selu()),
    'priv_resep_conv_5x5_htanh': lambda C, stride, affine:
        PrivResSepConv(C, C, 5, stride, 2, htanh()),


}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ResSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ResSepConv, self).__init__()
        self.conv = SepConv(C_in, C_out, kernel_size, stride, padding, affine)
        self.res = Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


######################## For Privacy ########################
def GN(plane):
    return nn.GroupNorm(4, plane, affine=False)

def relu():
    return nn.ReLU()

def elu():
    return nn.ELU()

def tanh():
    return nn.Tanh()

def htanh():
    return nn.Hardtanh()

def sigmoid():
    return nn.Sigmoid()

def selu():
    return nn.SELU()


class PrivReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(PrivReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            relu(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            GN(C_out)
        )

    def forward(self, x):
        return self.op(x)


class PrivDilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, Act):
        super(PrivDilConv, self).__init__()
        self.op = nn.Sequential(
            Act,
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            GN(C_out)
        )

    def forward(self, x):
        return self.op(x)


class PrivSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False, groups=C_out),
            GN(C_out),
            Act,

            # Act,
            # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
            #           padding=padding, groups=C_in, bias=False),
            # nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            # GN(C_in),
            # Act,
            # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
            #           padding=padding, groups=C_in, bias=False),
            # nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            # GN(C_out)
        )

    def forward(self, x):
        x = self.op(x)
        return x


class PrivResSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivResSepConv, self).__init__()
        self.conv = PrivSepConv(C_in, C_out, kernel_size, stride, padding, Act)
        self.res = Identity() if stride == 1 \
                else PrivFactorizedReduce(C_in, C_out, Act)
        self.res = (self.res)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])


class PrivConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, 3, stride=1, padding=1, bias=False, ),
            GN(C_in),
            Act,

            # Act,
            # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
            #           padding=padding, bias=False),
            # GN(C_out),
            # Act,
            # nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
            #           padding=padding, bias=False),
            # GN(C_out)
        )

    def forward(self, x):
        x = self.op(x)
        return x


class PrivResConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, Act):
        super(PrivResConv, self).__init__()
        self.conv = PrivConv(C_in, C_out, kernel_size, stride, padding, Act)
        self.res = Identity() if stride == 1 \
                else PrivFactorizedReduce(C_in, C_out, Act)
        self.res = (self.res)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])


class PrivFactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, Act=None):
        super(PrivFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = Act
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2,
                                padding=0, bias=False)
        self.bn = GN(C_out)

    def forward(self, x):
        if self.relu is not None:
            x = self.relu(x)
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)

        out1 = self.conv_1(x)
        out2 = self.conv_2(x[:, :, 1:, 1:])

        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        return out


