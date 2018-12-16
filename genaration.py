import torch
from utils import *
import multiprocessing as mul
from librosa.output import write_wav

generator = torch.load('./Results/Models/Gs_nch-8_epoch-29.pth')

DEVICE = torch.device('cuda:0')
z = hypersphere(torch.randn(10, 8 * 32, device=DEVICE))
c_0 = torch.zeros(10, 3).to(DEVICE)
c_0[:, 0] = 1
c_1 = torch.zeros(10, 3).to(DEVICE)
c_1[:, 1] = 1
c_2 = torch.zeros(10, 3).to(DEVICE)
c_2[:, 2] = 1


def generate(z, c, suffix):
    z_c_save = torch.cat((z, c), dim=1)
    z_c_save = torch.reshape(z_c_save, (z_c_save.shape[0], z_c_save.shape[1], 1, 1))
    with torch.no_grad():
        fake_mags = generator(z_c_save, 7)
    fake_mags = fake_mags.squeeze()
    fake_mags = fake_mags.to('cpu').numpy()
    fake_mag_list = [fake_mags[i] for i in range(z.shape[0])]
    pool = mul.Pool(10)
    audio_list = pool.map(reconstruct_from_magnitude, fake_mag_list)
    for i in range(z.shape[0]):
        write_wav('./Generation/'+suffix+'_'+str(i)+'.wav', audio_list[i], 44100) 

generate(z, c_0, 'type0')
generate(z, c_1, 'type1')
generate(z, c_2, 'type2')
   
