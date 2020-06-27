import random
import torch
import numpy as np
import os
import json

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models.dcgan_model as dcgan
import sampler
from plot import plot_kde, plot_scatter
from opt import get_opt

opt = get_opt()
print(opt)
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

# ===================
# Directory settings
# ===================
if not(os.path.exists(opt.dir_out)):
    os.mkdir(opt.dir_out)
if not( os.path.exists(os.path.join(opt.dir_out, opt.exper_name))):
    os.mkdir(os.path.join(opt.dir_out, opt.exper_name))
if not(os.path.exists(opt.tensorboard_dir)):
    os.mkdir(opt.tensorboard_dir)

# =====================
# Visualation settings
# =====================
board_train = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.exper_name))
# board_test = SummaryWriter( log_dir = os.path.join(opt.tensorboard_dir, opt.exper_name + "_test") )

# =======================
# weights initialization
# =======================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ======
# model
# ======
generator = dcgan.Generator(nz=opt.nz, nc=2).to(device)
discriminator = dcgan.Discriminator(nc=2).to(device)

# Handle multi-gpu if desired
if(cuda == True) and (opt.ngpu > 1):
    generator = nn.DataParallel(generator, list(range(opt.ngpu)))
    discriminator = nn.DataParallel(discriminator, list(range(opt.ngpu)))

# Apply the weights_init function to randomly initialize all weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# ==========
# optimizer
# ==========
if( opt.optimizer == "adam"):  
    optimizer_D = optim.Adam(discriminator.parameters(), lr = opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizer_G = optim.Adam(generator.parameters(), lr = opt.lrG, betas=(opt.beta1, opt.beta2))
else:
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr = opt.lrD)
    optimizer_G = optim.RMSprop(generator.parameters(), lr = opt.lrG)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize BCELoss function
criterion = nn.BCELoss()

# ======
# train
# ======
iterations = 0
for epoch in range(opt.nepochs):
    for i in range(opt.iterate):
        generator.train()
        discriminator.train()

        iterations += opt.batchSize

        # Adversarial ground truths
        valid = torch.full((opt.batchSize,), real_label, device=device)
        fake = torch.full((opt.batchSize,), fake_label, device=device)
        
        # sample from data distribution
        samples_ture = Variable(Tensor(sampler.gaussian_mixture_double_circle(batchsize=opt.batchSize, num_cluster=opt.num_mixture, scale=opt.scale, std=0.2))).to(device)

        for param in discriminator.parameters():
            param.requires_grad = True

        # ====================
        # Train the discriminator
        # ====================
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batchSize, opt.nz)).astype(np.float32))).to(device)

        # Generate a batch of images
        with torch.no_grad():
            samples_fake = generator(z)
            
        # Adversarial loss
        lossD_real = criterion(discriminator(samples_ture), valid)
        lossD_fake = criterion(discriminator(samples_fake.detach()), fake)
        lossD_real.backward()
        lossD_fake.backward()
        lossD = lossD_real + lossD_fake
        # lossD.backward()
        optimizer_D.step()

        # ====================
        # Train the generator 
        # ====================
        for param in discriminator.parameters():
            param.requires_grad = False

        z = Variable(Tensor(np.random.normal(0, 1, (opt.batchSize, opt.nz)).astype(np.float32))).to(device)

        optimizer_G.zero_grad()

        # Generate a batch of images
        samples_fake = generator(z)
        # Adversarial loss
        lossG = criterion(discriminator(samples_fake), valid)
        lossG.backward()
        optimizer_G.step()

        # ====================
        # Save to tensorborad
        # ====================
        if(i == 0 or (i % opt.n_display_step == 0)):
            board_train.add_scalar('Generater/loss_G', lossG.item(), iterations)
            board_train.add_scalar('Discriminator/loss_D', lossD.item(), iterations)
            board_train.add_scalar('Discriminator/loss_D_real', lossD_real.item(), iterations)
            board_train.add_scalar('Discriminator/loss_D_fake', lossD_fake.item(), iterations)
            # Monitor trainnig progresses
            print("epoch={}, iters={}, loss_G={:.5f}, loss_C={:.5f}".format(epoch, iterations, lossG, lossD))
        
    # ============
    # Save images
    # ============
    generator.eval()
    z_fixed = Variable(Tensor(np.random.normal(0, 1, (10000, opt.nz)).astype(np.float32))).to(device)
    with torch.no_grad():
        samples_fake = generator(z_fixed)
    
    plot_scatter(samples_fake.cpu().numpy(), dir=os.path.join(opt.dir_out, opt.exper_name), filename="scatter_epoches{}".format(epoch))
    plot_kde(samples_fake.cpu().numpy(), dir=os.path.join(opt.dir_out, opt.exper_name), filename="kde_epoches{}".format(epoch))