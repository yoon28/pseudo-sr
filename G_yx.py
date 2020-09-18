import torch
import torch.nn as nn

import common

class G_yx(nn.Module):
    def __init__(self, n_feat=64, n_resblock=6, bn=True):
        super(G_yx, self).__init__()
        leaky_neg = 0.2
        filter_size = 5
        z_channel = 6
        if bn:
            self.in_img = nn.Sequential(
                common.default_conv(3, n_feat//2, filter_size),
                nn.BatchNorm2d(n_feat//2),
                nn.LeakyReLU(leaky_neg),
                common.default_conv(n_feat//2, n_feat, filter_size),
                nn.BatchNorm2d(n_feat)
            )
        else:
            self.in_img = nn.Sequential(
                common.default_conv(3, n_feat//2, filter_size),
                nn.LeakyReLU(leaky_neg),
                common.default_conv(n_feat//2, n_feat, filter_size),
            )
        
        self.in_z = nn.Sequential(
            nn.ConvTranspose2d(1, z_channel, 2, 2, 0, 0), # 8 -> 16
            nn.LeakyReLU(leaky_neg),
            nn.ConvTranspose2d(z_channel, z_channel, 2, 2, 0, 0), # 16- > 32
        )
        self.merge = nn.Conv2d(n_feat + z_channel, n_feat, 1, 1, 0)
        resblocks = [
            common.ResBlock(common.default_conv, n_feat, filter_size, bn=bn, act=nn.LeakyReLU(leaky_neg)) \
            for _ in range(n_resblock)]
        self.res_blocks = nn.Sequential(*resblocks)
        self.fusion = nn.Sequential(
            common.default_conv(n_feat, n_feat//2, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//2, n_feat//4, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//4, 3, 1)
        )

    def forward(self, x, z=None):
        out_x = self.in_img(x)
        out_z = self.in_z(z)
        out = self.merge(torch.cat((out_x, out_z), dim=1))
        out = self.res_blocks(out)
        out = self.fusion(out)
        return out

if __name__ == "__main__":
    model = G_yx(n_feat=32, n_resblock=5, bn=False)
    print(model)

    X = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    Z = torch.rand(2, 1, 8, 8, dtype=torch.float32)
    Y = model(X, Z)
    print(X.shape, Y.shape)
