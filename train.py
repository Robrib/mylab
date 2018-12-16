import argparse
import os
import copy


import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import torch

from model import *
# from progressBar import printProgressBar
from utils import *

from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./dataset/training_data_512.npy', help='directory containing the data')
parser.add_argument('--outd', default='Results', help='directory to save results')
parser.add_argument('--outf', default='Spectograms', help='folder to save synthetic images')
parser.add_argument('--outl', default='Losses', help='folder to save Losses')
parser.add_argument('--outm', default='Models', help='folder to save models')

parser.add_argument('--nepoch', type=int, default=8, help='number of epoches')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchsize', type=int, default=16, help='batch size during the training')
parser.add_argument('--nch', type=int, default=16, help='base number of channel for networks')
parser.add_argument('--conditionsize', type=int, default=3, help='size of data c')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')

parser.add_argument('--n_iter', type=int, default=1, help='number of epochs to train before changing the progress')
parser.add_argument('--lambdaGP', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--gamma', type=float, default=1, help='gamma for gradient penalty')
parser.add_argument('--e_drift', type=float, default=0.01, help='epsilon drift for discriminator loss')
parser.add_argument('--savespectrograms', type=int, default=1, help='number of epochs between saving spectrogram examples')
parser.add_argument('--savenum', type=int, default=16, help='number of examples images to save')
parser.add_argument('--savemodel', type=int, default=1, help='number of epochs between saving models')
parser.add_argument('--savemaxsize', action='store_true', help='save sample images at max resolution instead of real resolution')
parser.add_argument('--verbose', action='store_true', help='show training progression')

opt = parser.parse_args()
print(opt)

DEVICE = torch.device('cuda:0')
MAX_RES = 7 # 8 for 1024x1024 output

x_train = np.load(opt.data)
dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train))


# creating output folders
if not os.path.exists(opt.outd):
    os.makedirs(opt.outd)
for f in [opt.outf, opt.outl, opt.outm]:
    if not os.path.exists(os.path.join(opt.outd, f)):
        os.makedirs(os.path.join(opt.outd, f))

# Model creation and init
G = torch.nn.DataParallel(Generator(max_res=MAX_RES, nch=opt.nch, nc=1, bn=opt.BN, ws=opt.WS, pn=opt.PN, c_size=opt.conditionsize).to(DEVICE), device_ids=[0, 1])
D = torch.nn.DataParallel(Discriminator(max_res=MAX_RES, nch=opt.nch, nc=1, bn=opt.BN, ws=opt.WS).to(DEVICE), device_ids=[0, 1])
Q = torch.nn.DataParallel(Q(max_res=MAX_RES, nch=opt.nch, nc=1, bn=opt.BN, ws=opt.WS).to(DEVICE), device_ids=[0, 1])
if not opt.WS:
    # weights are initialized by WScale layers to normal if WS is used
    G.apply(weights_init)
    D.apply(weights_init)
    Q.apply(weights_init)
Gs = copy.deepcopy(G).to(DEVICE)

optimizerG = Adam(G.parameters(), lr=1e-3, betas=(0, 0.99))
optimizerD = Adam(D.parameters(), lr=1e-3, betas=(0, 0.99))
optimizerQ = Adam(Q.parameters(), lr=1e-3, betas=(0, 0.99))

GP = GradientPenalty(opt.batchsize, opt.lambdaGP, opt.gamma, device=DEVICE)

epoch = 0
global_step = 0
total = 2
d_losses = np.array([])
d_losses_W = np.array([])
g_losses = np.array([])
q_losses = np.array([])
P = Progress(opt.n_iter, MAX_RES)

# z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))


# Creation of DataLoader
data_loader = DataLoader(dataset,
                         batch_size=opt.batchsize,
                         shuffle=True,
                         num_workers=opt.workers,
                         drop_last=True,
                         pin_memory=True)

while epoch < opt.nepoch:
    # t0 = time()
    print('# ============= Start epoch {:2d} =============#'.format(epoch+1))
    lossEpochG = []
    lossEpochD = []
    lossEpochD_W = []
    lossEpochQ = []

    total = len(data_loader)

    for i, (spectrograms, ) in enumerate(data_loader):
        t_start = time()
        P.progress(epoch, i + 1, total) # 
        global_step += 1

        # Build mini-batch
        spectrograms = spectrograms.to(DEVICE)
        spectrograms = P.resize(spectrograms)

        # ============= Train the discriminator =============#

        # zeroing gradients in D
        D.zero_grad()
        # compute fake spectrograms with G
        z = hypersphere(torch.randn(opt.batchsize, opt.nch * 32, device=DEVICE))
        c = sample_c_batch(opt.batchsize, opt.conditionsize, device=DEVICE)
        z_c = torch.cat((z, c), dim=1)
        z_c = torch.reshape(z_c, (z_c.shape[0], z_c.shape[1], 1, 1))
        with torch.no_grad():
            fake_spectrograms = G(z_c, P.p)
        del z_c, z, c

        # compute scores for real spectrograms
        D_real = D(spectrograms, P.p)
        D_realm = D_real.mean()

        # compute scores for fake spectrograms
        D_fake = D(fake_spectrograms, P.p)
        D_fakem = D_fake.mean()

        # compute gradient penalty for WGAN-GP as defined in the article
        gradient_penalty = GP(D, spectrograms.data, fake_spectrograms.data, P.p)
        del spectrograms
        del fake_spectrograms

        # prevent D_real from drifting too much from 0
        drift = (D_real ** 2).mean() * opt.e_drift

        # Backprop + Optimize
        d_loss = D_fakem - D_realm
        d_loss_W = d_loss + gradient_penalty + drift
        del D_real, D_realm, D_fake, D_fakem, gradient_penalty, drift


        d_loss_W.backward()
        optimizerD.step()

        lossEpochD.append(d_loss.item())
        lossEpochD_W.append(d_loss_W.item())

	# for saving memory

        # =============== Train the generator ===============#

        G.zero_grad()

        z = hypersphere(torch.randn(opt.batchsize, opt.nch * 32, device=DEVICE))
        c = sample_c_batch(opt.batchsize, opt.conditionsize, device=DEVICE)
        z_c = torch.cat((z, c), dim=1)
        z_c = torch.reshape(z_c, (z_c.shape[0], z_c.shape[1], 1, 1))
        fake_spectrograms = G(z_c, P.p)
        del z_c, z, c

        # compute scores with new fake spectrograms
        G_fake = D(fake_spectrograms, P.p)
        del fake_spectrograms
        G_fakem = G_fake.mean()
        # no need to compute D_real as it does not affect G
        g_loss = -G_fakem
        del G_fake, G_fakem

        # Optimize
        g_loss.backward()
        optimizerG.step()

        lossEpochG.append(g_loss.item())
	

        # update Gs with exponential moving average
        exp_mov_avg(Gs, G, alpha=0.999, global_step=global_step)
        
        # ============= Train the Q =============#

        # zeroing gradients in Q
        Q.zero_grad()
        
        # compute fake spectrograms with G
        z = hypersphere(torch.randn(opt.batchsize, opt.nch * 32, device=DEVICE))
        c = sample_c_batch(opt.batchsize, opt.conditionsize, device=DEVICE)
        z_c = torch.cat((z, c), dim=1)
        z_c = torch.reshape(z_c, (z_c.shape[0], z_c.shape[1], 1, 1))
        with torch.no_grad():
            fake_spectrograms = G(z_c, P.p)
        del z_c, z
 
        # compute Q outputs for fake spectrograms
        Q_fake = Q(fake_spectrograms, P.p)
        del fake_spectrograms
        q_loss = F.cross_entropy(Q_fake, torch.max(c, 1)[1])
        del Q_fake, c

        # Optimize
        q_loss.backward()
        optimizerQ.step()

        lossEpochQ.append(q_loss.item())
        

        t_end = time()
        time_batch = t_end - t_start
        
        if opt.verbose:
            if i < 1:
                print("epoch |   batch   |  time  |   - D_loss     |        G_loss  |      Q_loss" )
            print("  {:2d}  | {:4d}/{:4d} | {:6.2f} | {:14.6f} | {:14.6f} | {:14.6f}".format(epoch+1, i+1, total+1, time_batch, -d_loss, g_loss, q_loss))

    d_losses = np.append(d_losses, lossEpochD)
    d_losses_W = np.append(d_losses_W, lossEpochD_W)
    g_losses = np.append(g_losses, lossEpochG)
    q_losses = np.append(q_losses, lossEpochQ)

    np.save(os.path.join(opt.outd, opt.outl, 'd_losses.npy'), d_losses)
    np.save(os.path.join(opt.outd, opt.outl, 'd_losses_W.npy'), d_losses_W)
    np.save(os.path.join(opt.outd, opt.outl, 'g_losses.npy'), g_losses)
    np.save(os.path.join(opt.outd, opt.outl, 'q_losses.npy'), q_losses)

    if not (epoch + 1) % opt.savespectrograms:
        # Save sampled spectrograms with Gs
        Gs.eval()
        z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, device=DEVICE))
        c_save = sample_c_batch(opt.savenum, opt.conditionsize, device=DEVICE)
        z_c_save = torch.cat((z_save, c_save), dim=1)
        z_c_save = torch.reshape(z_c_save, (z_c_save.shape[0], z_c_save.shape[1], 1, 1))

        with torch.no_grad():
            fake_spectrograms = Gs(z_c_save, P.p)
            if opt.savemaxsize:
                if fake_spectrograms.size(-1) != 4 * 2 ** MAX_RES:
                    fake_spectrograms = F.upsample(fake_spectrograms, 4 * 2 ** MAX_RES)
        save_image(fake_spectrograms,
                   os.path.join(opt.outd, opt.outf, f'fake_spectrograms-{epoch:04d}-p{P.p:.2f}.png'),
                   nrow=8, pad_value=0,
                   normalize=True, range=(-1, 1))

    if P.p <= P.pmax and not epoch % opt.savemodel:
        torch.save(G, os.path.join(opt.outd, opt.outm, f'G_nch-{opt.nch}_epoch-{epoch}.pth'))
        torch.save(D, os.path.join(opt.outd, opt.outm, f'D_nch-{opt.nch}_epoch-{epoch}.pth'))
        torch.save(Gs, os.path.join(opt.outd, opt.outm, f'Gs_nch-{opt.nch}_epoch-{epoch}.pth'))
        torch.save(Q, os.path.join(opt.outd, opt.outm, f'Q_nch-{opt.nch}_epoch-{epoch}.pth'))

    epoch += 1
