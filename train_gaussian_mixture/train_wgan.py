import random
import torch
import numpy as np
import os
import json

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd

from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models.wgan_model as wgan
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
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# =========================
# Compute gradient penalty
# =========================
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates.view(d_interpolates.size(0), -1),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ===============================
# Compute W-div gradient penalty
# ===============================
def compute_div_gp(D, real_samples, fake_samples):
    real_interpolates = real_samples.requires_grad_(True)
    real_validity = D(real_interpolates)
    real_grad_out = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    real_grad = autograd.grad(
        outputs=real_validity.view(real_validity.size(0), -1),
        inputs=real_interpolates,
        grad_outputs=real_grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (opt.p / 2)

    fake_interpolates = fake_samples.requires_grad_(True)
    fake_validity = D(fake_interpolates)
    fake_grad_out = Variable(Tensor(fake_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake_grad = autograd.grad(
        outputs=fake_validity.view(fake_validity.size(0), -1),
        inputs=fake_interpolates,
        grad_outputs=fake_grad_out, 
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (opt.p / 2)

    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * opt.k / 2

    return div_gp

# ==========
# models
# ==========
generator = wgan.Generator(nz=opt.nz, nc=opt.nc).to(device)
discriminator = wgan.Discriminator(nc=opt.nc).to(device)

# ==========
# optimizer
# ==========
if( opt.optimizer == "rmsprop"):  
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr = opt.lrD)
    optimizer_G = optim.RMSprop(generator.parameters(), lr = opt.lrG)
else:
    optimizer_D = optim.Adam(discriminator.parameters(), lr = opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizer_G = optim.Adam(generator.parameters(), lr = opt.lrG, betas=(opt.beta1, opt.beta2))

# Handle multi-gpu if desired
if(cuda == True) and (opt.ngpu > 1):
    generator = nn.DataParallel(generator, list(range(opt.ngpu)))
    discriminator = nn.DataParallel(discriminator, list(range(opt.ngpu)))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ======
# train
# ======
iterations = 0
for epoch in range(opt.nepochs):
    for i in range(opt.iterate):
        generator.train()
        discriminator.train()

        iterations += opt.batchSize
        
        for param in discriminator.parameters():
            param.requires_grad = True

        for n in range(opt.n_critic):
            # ====================
            # Train the discriminator
            # ====================
            optimizer_D.zero_grad()

            # sample from data distribution
            samples_ture = Variable(Tensor(sampler.gaussian_mixture_double_circle(batchsize=opt.batchSize, num_cluster=opt.num_mixture, scale=opt.scale, std=0.2))).to(device)

            # sample from generator
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batchSize, opt.nz)).astype(np.float32))).to(device)

            # Generate a batch of images
            with torch.no_grad():
                samples_fake = generator(z)

            # Adversarial loss
            real_validity = discriminator(samples_ture / opt.scale)
            fake_validity = discriminator(samples_fake.detach() / opt.scale)
            if(opt.model == "wgan-gp"):
                gradient_penalty = compute_gradient_penalty(discriminator, samples_ture.data, samples_fake.data)
                lossD = -torch.sum(real_validity - fake_validity) / opt.batchSize + opt.lambda_gp * gradient_penalty
                lossD.backward()
                optimizer_D.step()
            elif(opt.model == "wgan-div"):
                div_gp = compute_div_gp(discriminator, samples_ture.data, samples_fake.data)
                lossD = -torch.sum(real_validity - fake_validity) / opt.batchSize + div_gp
                lossD.backward()
                optimizer_D.step()
            else:
                # lossD =  -lossD_real + lossD_fake
                lossD = -torch.sum(real_validity - fake_validity) / opt.batchSize
                lossD.backward()
                optimizer_D.step()
                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

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
        lossG = - torch.sum(discriminator(samples_fake / opt.scale) / opt.batchSize)
        lossG.backward()
        optimizer_G.step()

        # ====================
        # Save to tensorborad
        # ====================
        if(i == 0 or (i % opt.n_display_step == 0)):
            board_train.add_scalar('Generater/loss_G', lossG.item(), iterations)
            board_train.add_scalar('Discriminator/loss_D', lossD.item(), iterations)
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