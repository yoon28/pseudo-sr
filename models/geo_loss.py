import torch
import torch.nn as nn

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
    random_seed = 2020
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
    x = torch.cat((x, x, x, x), dim=0)
    print(x)
    crit = nn.L1Loss()
    #net = nn.Identity()
    net = nn.Conv2d(1, 1, 3, 1, 1)
    net.weight.data.normal_()
    net.bias.data.fill_(0)
    import torch.optim as optim
    sgd = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for i in range(10000):
        sgd.zero_grad()
        y = net(x)
        out = geometry_ensemble(net, x, flip_dir="H")
        loss = crit(out, y)
        loss.backward()
        sgd.step()
        print(f"{i}: {loss.item():.8f}")
    #print(out)
