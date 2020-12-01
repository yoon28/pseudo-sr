import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        ## for x16 yoon
        #self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        ## yoon

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class RRDBNet_H2L_x2_D1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, n_noise=128, in_hw=512):
        super(RRDBNet_H2L_x2_D1, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.in_hw = in_hw
        self.noise_fc = nn.Linear(n_noise, in_hw * in_hw)

        self.conv_first = nn.Conv2d(in_nc, nf, 5, 2, 2, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.LRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x, z):
        fz = self.relu(self.noise_fc(z))
        fz = fz.view(-1, 1, self.in_hw, self.in_hw)
        fxz = torch.cat((x, fz), 1)

        fea = self.lrelu(self.conv_first(fxz))
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(fea)
        out = self.conv_last(self.lrelu(self.LRconv(fea)))

        return out

class MyCNN_H2L_x2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, in_hw=128, device="cuda"):
        super(MyCNN_H2L_x2, self).__init__()
        self.in_hw = in_hw
        self.out_hw = int(in_hw // 2)

        self.conv1 = nn.Conv2d(in_nc, 64, 7, 2, 3, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(64, 64, 1, 1, 0, bias=True)
        self.conv6 = nn.Conv2d(64, out_nc, 1, 1, 0, bias=True)
        self.activ = nn.ReLU() # nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.noise = nn.Parameter(torch.zeros([1, out_nc, self.out_hw, self.out_hw], requires_grad=True, device=device))

        self.apply(weight_init_G)

    def forward(self, x, z=None):
        feat_1 = self.activ(self.conv1(x))
        feat = self.activ(self.conv2(feat_1))
        feat = self.activ(self.conv3(feat))
        feat = self.activ(self.conv4(feat))
        feat = self.conv5(feat) + feat_1
        feat = self.activ(feat)
        #feat = self.conv6(feat) + self.noise
        feat = self.activ(self.conv6(feat)) + self.noise
        return feat

def weight_init_G(m):
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)

class MyCNN_H2L_x2_convNoise(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, in_hw=128, device="cuda"):
        super(MyCNN_H2L_x2_convNoise, self).__init__()
        self.in_hw = in_hw
        self.out_hw = int(in_hw // 2)

        self.conv1 = nn.Conv2d(in_nc, 64, 7, 2, 3, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2, bias=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(64, 64, 1, 1, 0, bias=True)
        self.conv6 = nn.Conv2d(64, out_nc, 1, 1, 0, bias=True)
        self.activ = nn.LeakyReLU(negative_slope=0.2, inplace=False) # nn.ReLU()
        # self.noise = nn.Parameter(torch.zeros([1, 3, self.out_hw, self.out_hw], requires_grad=True, device=device))
        self.noise = nn.Conv2d(out_nc, out_nc, 5, 1, 2, bias=True)

        self.apply(weight_init_G)

    def forward(self, x, z=None):
        feat_1 = self.activ(self.conv1(x))
        feat = self.activ(self.conv2(feat_1))
        feat = self.activ(self.conv3(feat))
        feat = self.activ(self.conv4(feat))
        feat = self.conv5(feat) + feat_1
        feat = self.activ(feat)
        # feat = self.conv6(feat) + self.noise
        feat = self.noise(self.activ(self.conv6(feat)))
        return feat

    def regularize(self):
        pass

class RRDBNet_H2L_x2_double(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, in_hw=512):
        super(RRDBNet_H2L_x2_double, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.in_hw = in_hw
        #self.noise_fc = nn.Linear(n_noise, in_hw * in_hw)

        self.conv_first = nn.Conv2d(in_nc, nf, 5, 2, 2, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.LRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        #self.activ_out = nn.Tanh() # nn.Sigmoid()

    def forward(self, x, z=None):
        #fz = self.noise_fc(z)
        #fz = fz.view(-1, 1, self.in_hw, self.in_hw)
        if z:
            fxz = torch.cat((x, z), 1)
        else:
            fxz = x

        fea = self.conv_first(fxz)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.avgpool(fea)
        fea = self.lrelu(self.downconv1(fea))
        fea = nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
        out = self.conv_last(self.lrelu(self.LRconv(fea)))
        #out = self.activ_out(out)

        return out

class RRDBNet_H2L_x2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, in_hw=512):
        super(RRDBNet_H2L_x2, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.in_hw = in_hw
        #self.noise_fc = nn.Linear(n_noise, in_hw * in_hw)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.LRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x, z):
        #fz = self.noise_fc(z)
        #fz = fz.view(-1, 1, self.in_hw, self.in_hw)
        fxz = torch.cat((x, z), 1)

        fea = self.conv_first(fxz)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.avgpool(fea)
        fea = self.lrelu(self.downconv1(fea))
        out = self.conv_last(self.lrelu(self.LRconv(fea)))

        return out

class RRDBNet_L2H_x2(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, in_hw=256, pretrained=None):
        super(RRDBNet_L2H_x2, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.in_hw = in_hw

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        #self.activ_out = nn.Tanh() # nn.Sigmoid()

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        #out = self.activ_out(out)

        return out

if __name__ == "__main__":
    im_size = 128
    z_size = im_size
    X = torch.randn([4, 3, im_size, im_size], dtype=torch.float32).cuda()
    Z = torch.randn(4, 1, im_size, im_size).cuda()
    net = MyCNN_H2L_x2().cuda() # RRDBNet_H2L_x2_double(in_nc=4, out_nc=3, nf=64, nb=23, in_hw=im_size).cuda()
    Y = net(X, Z)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    print("X", X.size())
    print("Y", Y.size())
    print("finished")
