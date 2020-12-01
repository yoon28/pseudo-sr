from multiprocessing.connection import deliver_challenge
import os
import random

import cv2
import numpy as np
import argparse
from numpy.lib.npyio import load
from torch.nn.modules import loss
from yacs.config import CfgNode

import torch
from torch._C import device, parse_type_comment
import torch.multiprocessing as mp
import torch.distributed as dist

from tools.pseudo_face_data import faces_data
from models.face_trainer import Face_Model


main_parse = argparse.ArgumentParser()
main_parse.add_argument("yaml", type=str)
main_parse.add_argument("--port", type=int, default=2357)
main_args = main_parse.parse_args()
with open(main_args.yaml, "rb") as cf:
    CFG = CfgNode.load_cfg(cf)
    CFG.freeze()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_train_loader(trainset, world_size, batch_size):
    if world_size <= 1:
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=CFG.DATA.NUM_WORKERS, pin_memory=True)
    elif world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=CFG.DATA.NUM_WORKERS, pin_memory=True)

def main(rank, world_size, cpu=False):
    if cpu:
        rank = torch.device("cpu")
    elif world_size > 1:
        setup(rank, world_size)
    last_device = world_size - 1
    batch_per_gpu = CFG.SR.BATCH_PER_GPU
    start_ep = CFG.OPT.START_EPOCH
    end_ep = CFG.OPT.EPOCH

    model = Face_Model(rank, CFG, world_size > 1)

    trainset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "LOW/wider_lnew"), data_hr=os.path.join(CFG.DATA.FOLDER, "HIGH"), img_range=CFG.DATA.IMG_RANGE)
    loader = get_train_loader(trainset, world_size, batch_per_gpu)
    testset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "testset"), data_hr=None, b_train=False, shuffle=False)

    for ep in range(start_ep, end_ep):
        model.mode_selector("train")
        for b, batch in enumerate(loader):
            lrs = batch["lr"].to(rank)
            hrs = batch["hr"].to(rank)
            zs = batch["z"].to(rank)
            hr_downs = batch["hr_down"].to(rank)
            losses = model.train_step(hrs, lrs, hr_downs, zs)
            info = f"{model.n_iter}/({ep}):"
            for itm in losses.items():
                info += f", {itm[0]}={itm[1]:.3f}"
            print(info + "\r", end="")

        model.lr_decay_step()
        if ep % 1 == 0 and rank == last_device:
            print("\nTesting")
            model.mode_selector("eval")
            for b in range(len(testset)):
                if b > 100: break
                lr = testset[b]["lr"].unsqueeze(0).to(rank)
                with torch.no_grad():
                    y = model.G_xy(lr)
                    sr = model.U(y)
                y = y.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
                sr = sr.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
                lrimg = lr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                lrimg = np.around(lrimg[:, :, ::-1] * 255).astype(np.uint8)
                y_im = np.around(np.clip(y[:, :, ::-1], a_min=0, a_max=1) * 255).astype(np.uint8)
                sr_im = np.around(np.clip(sr[:, :, ::-1], a_min=0, a_max=1) * 255).astype(np.uint8)
                cv2.imwrite("temp/{:04d}_lr.png".format(b), lrimg)
                cv2.imwrite("temp/{:04d}_y.png".format(b), y_im)
                cv2.imwrite("temp/{:04d}_sr.png".format(b), sr_im)
    if world_size > 1:cleanup()

if __name__ == "__main__":
    random_seed = CFG.EXP.SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    
    if n_gpus <= 1:
        print("single proc.")
        main(0, n_gpus, cpu=(n_gpus == 0))
    elif n_gpus > 1:
        print(f"multi-gpu: {n_gpus}")
        mp.spawn(main, nprocs=n_gpus, args=(n_gpus, False), join=True)
    print("fin.")
