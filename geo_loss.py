import torch
import torch.nn as nn
import numpy as np

def geometry_ensemble(net, img, flip_dir="H"):
    assert len(img.shape) == 4
    assert flip_dir in ["H", "V", None]
    if flip_dir == "H":
        flip_axes = [3]
    elif flip_dir == "V":
        flip_axes = [2]
    imgs = []
    for r in range(4):
        imgs.append(torch.rot90(img, r, [2, 3]))
    if flip_dir:
        flips = []
        for im in imgs:
            flips.append(torch.flip(im, flip_axes))
        imgs.extend(flips)
    outs = []
    for r, im in enumerate(imgs):
        temp = net(im)
        if r < 4:
            outs.append(torch.rot90(temp, -r, [2, 3]))
        else:
            temp2 = torch.flip(temp, flip_axes)
            outs.append(torch.rot90(temp2, -(r%4), [2, 3]))
    for i in range(1, len(outs)):
        outs[0] += outs[i]
    return outs[0] / len(outs)


if __name__ == "__main__":
    x = torch.arange(8, dtype=torch.float).view(1, 2, 2, 2)
    print(x)
    net = nn.Identity()
    out = geometry_ensemble(net, x, flip_dir="H")
    print(out)