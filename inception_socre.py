import argparse
import random
import torch
import numpy as np

from torch.autograd import Variable

from utils import load_checkpoint , load_checkpoint_wo_step
from utils import inception_score

import models.wgan_model as wgan

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help="path to generator")
parser.add_argument('--dataset', required=True, help="cifar-10 | folder | mnist")
parser.add_argument('--device', default="gpu", help="(CPU or GPU)")
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--num_data', type=int, default=1024)
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--imageSize', type=int, default=64, help='output image size')
opt = parser.parse_args()

# =======
# device
# =======
cuda = True if torch.cuda.is_available() else False
if(opt.device == "gpu"):
    if(cuda):
        device = torch.device("cuda")
    else:
        device = torch.device( "cpu" )
else:
    device = torch.device( "cpu" )

# ======
# model
# ======
generator = wgan.Generator(img_size=opt.imageSize, nz=opt.nz, nc=opt.nc, ngf=opt.ngf).to(device)

# ===========
# load model
# ===========
G = load_checkpoint_wo_step(generator, device, opt.model)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ====================
# calc inception score
# ===================
score_z = Variable(Tensor(np.random.normal(0, 1, (opt.num_data, opt.nz, 1, 1)))).to(device)
score_imgs = G(score_z)
score_mean, score_std = inception_score(imgs=score_imgs, cuda=cuda, batch_size=opt.batchSize, resize=True, splits=1)
print(score_mean, score_std)