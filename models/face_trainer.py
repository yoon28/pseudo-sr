import torch
from torch import autograd
from torch._C import device
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import cv2
import numpy as np

from models.rcan import make_cleaning_net, make_SR_net
from models.generators import TransferNet
from models.discriminators import NLayerDiscriminator
from models.losses import GANLoss
from models.geo_loss import geometry_ensemble

class Face_Model():
    def __init__(self, device, cfg, use_ddp=False):
        self.device = device
        self.idt_input_clean = cfg.OPT_CYC.IDT_INPUT == "clean"
        rgb_range = cfg.DATA.IMG_RANGE
        rgb_mean_point = (0.5, 0.5, 0.5) if cfg.DATA.IMG_MEAN_SHIFT else (0, 0, 0)
        self.G_xy = make_cleaning_net(rgb_range=rgb_range, rgb_mean=rgb_mean_point).to(device)
        self.G_yx = TransferNet(rgb_range=rgb_range, rgb_mean=rgb_mean_point).to(device)
        self.U = make_SR_net(rgb_range=rgb_range, rgb_mean=rgb_mean_point, scale_factor=2).to(device)
        self.D_x = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity).to(device)
        self.D_y = NLayerDiscriminator(3, scale_factor=1, norm_layer=nn.Identity).to(device)
        self.D_sr = NLayerDiscriminator(3, scale_factor=cfg.SR.SCALE, norm_layer=nn.Identity).to(device)
        if use_ddp:
            self.G_xy = DDP(self.G_xy, device_ids=[device])
            self.G_yx = DDP(self.G_yx, device_ids=[device])
            self.U = DDP(self.U, device_ids=[device])
            self.D_x = DDP(self.D_x, device_ids=[device])
            self.D_y = DDP(self.D_y, device_ids=[device])
            self.D_sr = DDP(self.D_sr, device_ids=[device])

        self.opt_Gxy = optim.Adam(self.G_xy.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Gyx = optim.Adam(self.G_yx.parameters(), lr=cfg.OPT_CYC.LR_G, betas=cfg.OPT_CYC.BETAS_G)
        self.opt_Dx = optim.Adam(self.D_x.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_Dy = optim.Adam(self.D_y.parameters(), lr=cfg.OPT_CYC.LR_D, betas=cfg.OPT_CYC.BETAS_D)
        self.opt_U = optim.Adam(self.U.parameters(), lr=cfg.OPT_SR.LR_G, betas=cfg.OPT_SR.BETAS_G)
        self.opt_Dsr = optim.Adam(self.D_sr.parameters(), lr=cfg.OPT_SR.LR_D, betas=cfg.OPT_SR.BETAS_D)

        self.lr_Gxy = optim.lr_scheduler.MultiStepLR(self.opt_Gxy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Gyx = optim.lr_scheduler.MultiStepLR(self.opt_Gyx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dx = optim.lr_scheduler.MultiStepLR(self.opt_Dx, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_Dy = optim.lr_scheduler.MultiStepLR(self.opt_Dy, milestones=cfg.OPT_CYC.LR_MILESTONE, gamma=cfg.OPT_CYC.LR_DECAY)
        self.lr_U = optim.lr_scheduler.MultiStepLR(self.opt_U, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)
        self.lr_Dsr = optim.lr_scheduler.MultiStepLR(self.opt_Dsr, milestones=cfg.OPT_SR.LR_MILESTONE, gamma=cfg.OPT_SR.LR_DECAY)

        self.nets = {"G_xy":self.G_xy, "G_yx":self.G_yx, "U":self.U, "D_x":self.D_x, "D_y":self.D_y, "D_sr":self.D_sr}
        self.optims = {"G_xy":self.opt_Gxy, "G_yx":self.opt_Gyx, "U":self.opt_U, "D_x":self.opt_Dx, "D_y":self.opt_Dy, "D_sr":self.opt_Dsr}
        self.lr_decays = {"G_xy":self.lr_Gxy, "G_yx":self.lr_Gyx, "U":self.lr_U, "D_x":self.lr_Dx, "D_y":self.lr_Dy, "D_sr":self.lr_Dsr}
        self.discs = ["D_x", "D_y", "D_sr"]
        self.gens = ["G_xy", "G_yx", "U"]

        self.n_iter = 0
        self.gan_loss = GANLoss("lsgan")
        self.l1_loss = nn.L1Loss()

        self.d_sr_weight = cfg.OPT_CYC.LOSS.D_SR_WEIGHT
        self.cyc_weight = cfg.OPT_CYC.LOSS.CYC_WEIGHT
        self.idt_weight = cfg.OPT_CYC.LOSS.IDT_WEIGHT
        self.geo_weight = cfg.OPT_CYC.LOSS.GEO_WEIGHT

    def net_grad_toggle(self, nets, need_grad):
        for n in nets:
            for p in self.nets[n].parameters():
                p.requires_grad = need_grad

    def mode_selector(self, mode="train"):
        if mode == "train":
            for n in self.nets:
                self.nets[n].train()
        elif mode in ["eval", "test"]:
            for n in self.nets:
                self.nets[n].eval()

    def lr_decay_step(self, shout=False):
        for n in self.lr_decays:
            self.lr_decays[n].step()

    def train_step(self, Ys, Xs, Yds, Zs):
        '''
        Ys: high resolutions
        Xs: low resolutions
        Yds: down sampled HR
        Zs: noises
        '''
        self.n_iter += 1
        loss_dict = dict()

        # forward
        fake_Xs = self.G_yx(Yds, Zs)
        rec_Yds = self.G_xy(fake_Xs)
        fake_Yds = self.G_xy(Xs)
        geo_Yds = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Yds) if self.idt_input_clean else fake_Yds
        sr_y = self.U(rec_Yds)
        sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (self.gan_loss(pred_real_Xs, True, True) + self.gan_loss(pred_fake_Xs, False, True)) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Yds = self.D_y(fake_Yds.detach())
        pred_real_Yds = self.D_y(Yds)
        loss_D_y = (self.gan_loss(pred_real_Yds, True, True) + self.gan_loss(pred_fake_Yds, False, True)) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        # D_sr
        pred_sr_x = self.D_sr(sr_x.detach())
        pred_sr_y = self.D_sr(sr_y.detach())
        loss_D_sr = (self.gan_loss(pred_sr_x, True, True) + self.gan_loss(pred_sr_y, False, True)) * 0.5
        self.opt_Dsr.zero_grad()
        loss_D_sr.backward()
        self.opt_Dsr.step()
        loss_dict["D_sr"] = loss_D_sr.item()

        self.net_grad_toggle(["D_x", "D_y", "D_sr"], False)
        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Yds = self.D_y(fake_Yds)
        pred_sr_y = self.D_sr(sr_y)
        loss_gan_Gxy = self.gan_loss(pred_fake_Yds, True, False)
        loss_idt_Gxy = self.l1_loss(idt_out, Yds) if self.idt_input_clean else self.l1_loss(idt_out, Xs)
        loss_cycle = self.l1_loss(rec_Yds, Yds)
        loss_geo = self.l1_loss(fake_Yds, geo_Yds)
        loss_sr_d = self.gan_loss(pred_sr_y, True, False)
        loss_total_gen = loss_gan_Gyx + loss_gan_Gxy + self.cyc_weight * loss_cycle + self.idt_weight * loss_idt_Gxy + self.geo_weight * loss_geo + self.d_sr_weight * loss_sr_d
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["sr_d_loss"] = loss_sr_d.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        self.opt_U.zero_grad()
        loss_U = self.l1_loss(self.U(rec_Yds.detach()), Ys)
        loss_U.backward()
        self.opt_U.step()
        loss_dict["U_loss"] = loss_U.item()
        return loss_dict

if __name__ == "__main__":
    print("fin.")
