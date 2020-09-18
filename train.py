import os, sys
import random
import numpy as np
import cv2

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from G_yx import G_yx
from discriminators import NLayerDiscriminator, VGGFeatureExtractor
from rcan import make_G_SR, make_G_cleaning
from rrdb import RRDBNet_L2H_x2

from geo_loss import geometry_ensemble
from pseudo_face_data import faces_data

import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed_num = 0
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    high_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
    low_folder = os.path.join(os.environ["DATA_TRAIN"], "LOW")
    test_folder = os.path.join(os.environ["DATA_TEST"])

    max_epoch = 50
    learn_rate = 5e-5
    batch_size = 32
    cyc_importance = 1
    idt_importance = 2
    geo_importance = 1
    hr_importance = 0.1
    mile_stones = [100000, 180000, 240000, 280000]

    G_degrade = G_yx().cuda()
    G_clean = make_G_cleaning().cuda()
    #G_SR = make_G_SR().cuda()
    G_SR = RRDBNet_L2H_x2(in_nc=3, out_nc=3, nf=64, nb=23, in_hw=32).cuda()
    D_x = NLayerDiscriminator(3, norm_layer=torch.nn.Identity, stride_input=1).cuda()
    D_y = NLayerDiscriminator(3, norm_layer=torch.nn.Identity, stride_input=1).cuda()
    D_SR = NLayerDiscriminator(3, norm_layer=torch.nn.Identity, stride_input=2).cuda()
    D_ESR = NLayerDiscriminator(3, norm_layer=torch.nn.Identity, stride_input=2).cuda()

    opt_G_degrade = optim.Adam(G_degrade.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    opt_G_clean = optim.Adam(G_clean.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    opt_G_SR = optim.Adam(G_SR.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    opt_D_x = optim.Adam(D_x.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    opt_D_y = optim.Adam(D_y.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    opt_D_SR = optim.Adam(D_SR.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    opt_D_ESR = optim.Adam(D_ESR.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    decay_lr = [torch.optim.lr_scheduler.MultiStepLR(opt_G_clean, milestones=mile_stones, gamma=0.5),
            torch.optim.lr_scheduler.MultiStepLR(opt_G_degrade, milestones=mile_stones, gamma=0.5),
            torch.optim.lr_scheduler.MultiStepLR(opt_D_x, milestones=mile_stones, gamma=0.5),
            torch.optim.lr_scheduler.MultiStepLR(opt_D_y, milestones=mile_stones, gamma=0.5),
            torch.optim.lr_scheduler.MultiStepLR(opt_D_SR, milestones=mile_stones, gamma=0.5)]

    data = faces_data(low_folder, high_folder)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    testset = faces_data(test_folder, shuffle=False)

    pix_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    vgg_loss = VGGFeatureExtractor(device=torch.device("cuda"))

    iterations = 0
    for ep in range(1, max_epoch+1):
        G_degrade.train()
        G_clean.train()
        G_SR.train()
        for i, batch in enumerate(loader):
            iterations += 1
            z = batch["z"].cuda()
            lr = batch["lr_upx2"].cuda()
            hr = batch["hr"].cuda()
            hr_y = batch["hr_down"].cuda()
            n_samples = z.size(0)

            # Discriminator for LR
            for D in [D_x, D_y, D_SR]:
                for p in D.parameters():
                    p.requires_grad = True
                D.zero_grad()
            for p in G_SR.parameters():
                p.requires_grad = False
            G_SR.zero_grad()
            x_fake = G_degrade(hr_y, z).detach()
            y_fake = G_clean(lr).detach()
            y_recon = G_clean(x_fake).detach()
            sr_x = G_SR(y_fake).detach()
            sr_y = G_SR(y_recon).detach()

            disc_y_real = D_y(hr_y)
            disc_y_fake = D_y(y_fake)
            disc_x_real = D_x(lr)
            disc_x_fake = D_x(x_fake)
            disc_sr_x = D_SR(sr_x)
            disc_sr_y = D_SR(sr_y)

            ones = torch.ones([n_samples, 1, disc_y_real.size(2), disc_y_real.size(3)], dtype=torch.float32).cuda()
            zeros = torch.zeros([n_samples, 1, disc_y_real.size(2), disc_y_real.size(3)], dtype=torch.float32).cuda()
            ones_sr = torch.zeros([n_samples, 1, disc_sr_x.size(2), disc_sr_x.size(3)], dtype=torch.float32).cuda()
            zeros_sr = torch.zeros([n_samples, 1, disc_sr_x.size(2), disc_sr_x.size(3)], dtype=torch.float32).cuda()

            err_D_y = (bce_loss(disc_y_real - torch.mean(disc_y_fake), ones) + bce_loss(disc_y_fake - torch.mean(disc_y_real), zeros)) * 0.5
            err_D_x = (bce_loss(disc_x_real - torch.mean(disc_x_fake), ones) + bce_loss(disc_x_fake - torch.mean(disc_x_real), zeros)) * 0.5
            err_D_SR = (bce_loss(disc_sr_x - torch.mean(disc_sr_y), ones_sr) + bce_loss(disc_sr_y - torch.mean(disc_sr_x), zeros_sr)) * 0.5

            err_D = err_D_SR + err_D_x + err_D_y
            err_D.backward()
            opt_D_x.step()
            opt_D_y.step()
            opt_D_SR.step()

            # Generator for LR
            for D in [D_x, D_y, D_SR]:
                for p in D.parameters():
                    p.requires_grad = False
                D.zero_grad()
            G_SR.zero_grad()
            G_clean.zero_grad()
            G_degrade.zero_grad()

            gen_x_fake = G_degrade(hr_y, z)
            gen_y_fake = G_clean(lr)
            recon_y = G_clean(gen_x_fake)
            geo_y_ensemble = geometry_ensemble(G_clean, lr)
            #idt_x = G_clean(lr)
            sr_x = G_SR(gen_y_fake)
            sr_y = G_SR(recon_y)

            disc_x_fake = D_x(gen_x_fake)
            disc_y_fake = D_y(gen_y_fake)
            disc_x_real = D_x(lr).detach() #
            disc_y_real = D_y(hr_y).detach() #
            disc_sr_x = D_SR(sr_x).detach() #
            disc_sr_y = D_SR(sr_y)

            err_adv_y = (bce_loss(disc_y_real - torch.mean(disc_y_fake), zeros) + bce_loss(disc_y_fake - torch.mean(disc_y_real), ones)) * 0.5
            err_adv_x = (bce_loss(disc_x_real - torch.mean(disc_x_fake), zeros) + bce_loss(disc_x_fake - torch.mean(disc_x_real), ones)) * 0.5
            err_adv_sr = (bce_loss(disc_sr_x - torch.mean(disc_sr_y), zeros_sr) + bce_loss(disc_sr_y - torch.mean(disc_sr_x), ones_sr)) * 0.5

            err_cyc = pix_loss(recon_y, hr_y)
            err_idt = pix_loss(gen_y_fake, lr) #pix_loss(idt_x, lr)
            err_geo = pix_loss(gen_y_fake, geo_y_ensemble)

            err_G = err_adv_y + err_adv_x + hr_importance * err_adv_sr + cyc_importance * err_cyc + idt_importance * err_idt + geo_importance + err_geo
            err_G.backward()
            opt_G_clean.step()
            opt_G_degrade.step()

            # SR network
            if ep > 1: # GAN
                D_ESR.zero_grad()
                disc_sr_real = D_ESR(hr)
                disc_sr_fake = D_ESR(sr_y.detach())

                err_D_ESR = (bce_loss(disc_sr_real - torch.mean(disc_sr_fake), ones_sr) + bce_loss(disc_sr_fake - torch.mean(disc_sr_real), zeros_sr)) * 0.5
                err_D_ESR.backward()
                opt_D_ESR.step()
                err_D_ESR_metric = err_D_ESR.item()

                for p in G_SR.parameters():
                    p.requires_grad = True
                G_SR.zero_grad()
                gen_sr_fake = G_SR(recon_y.detach())
                disc_sr_fake = D_ESR(gen_sr_fake)
                disc_sr_real = D_ESR(hr).detach() #
                vgg_sr_fake = vgg_loss(gen_sr_fake)
                vgg_sr_real = vgg_loss(hr).detach()
                gan_err = (bce_loss(disc_sr_real - torch.mean(disc_sr_fake), zeros_sr) + bce_loss(disc_sr_fake - torch.mean(disc_sr_real), ones_sr)) * 0.5
                pix_err = pix_loss(gen_sr_fake, hr)
                vgg_err = pix_loss(vgg_sr_fake, vgg_sr_real)

                err_SR = 5e-3 * gan_err + 1e-2 * pix_err + vgg_err
                err_SR.backward()
                opt_G_SR.step()
            else: # MSE
                for p in G_SR.parameters():
                    p.requires_grad = True
                G_SR.zero_grad()
                gen_sr_fake = G_SR(recon_y.detach())
                pix_err = pix_loss(gen_sr_fake, hr)
                vgg_sr_fake = vgg_loss(gen_sr_fake)
                vgg_sr_real = vgg_loss(hr).detach()
                vgg_err = pix_loss(vgg_sr_fake, vgg_sr_real)
                err_SR = pix_err + 0.1 * vgg_err
                err_SR.backward()
                opt_G_SR.step()
                err_D_ESR_metric = -1.23

            for decay in decay_lr:
                decay.step()
            current_lr = decay_lr[0].get_last_lr()[0]
            print("  ep={:3d}, iter={:6d}, loss_D={:.6f}, loss_G={:.6f}, loss_SR_D={:.6f}, loss_SR_G={:.6f}, lr={:.7f}\r ".format(ep, iterations, err_D.item(), err_G.item(), err_D_ESR_metric, err_SR.item(), current_lr), end=" ")

        if ep % 1 == 0:
            print("\ntesting....")
            G_SR.eval()
            G_clean.eval()
            G_degrade.eval()
            for i in range(len(testset)):
                if i > 100: break
                sample = testset[i]
                lr = sample["lr_upx2"].unsqueeze(0).cuda()
                with torch.no_grad():
                    y = G_clean(lr)
                    sr = G_SR(y)
                y = y.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
                sr = sr.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
                lr_im = lr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                lr_im = np.around(lr_im[:, :, ::-1]* 255).astype(np.uint8)
                y_im = np.around(np.clip(y[:, :, ::-1], a_min=0, a_max=1) * 255).astype(np.uint8)
                sr_im = np.around(np.clip(sr[:, :, ::-1], a_min=0, a_max=1) * 255).astype(np.uint8)
                cv2.imwrite("temp/{:04d}_lr.png".format(i), lr_im)
                cv2.imwrite("temp/{:04d}_y.png".format(i), y_im)
                cv2.imwrite("temp/{:04d}_sr.png".format(i), sr_im)
