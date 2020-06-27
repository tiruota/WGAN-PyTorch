import argparse
import random
import torch
import numpy as np
import os
import json
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models.dcgan_model as dcgan
from utils import save_checkpoint, load_checkpoint, save_checkpoint_wo_step
from utils import board_add_image, board_add_images
from utils import save_image_historys_gif
from utils import inception_score

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
if not(os.path.exists(opt.save_checkpoints_dir)):
    os.mkdir(opt.save_checkpoints_dir)
if not(os.path.exists(os.path.join(opt.save_checkpoints_dir, opt.exper_name))):
    os.mkdir( os.path.join(opt.save_checkpoints_dir, opt.exper_name))
if not(os.path.exists(os.path.join(opt.save_checkpoints_dir, opt.exper_name, "G")) ):
    os.mkdir( os.path.join(opt.save_checkpoints_dir, opt.exper_name, "G") )
if not(os.path.exists(os.path.join(opt.save_checkpoints_dir, opt.exper_name, "D")) ):
    os.mkdir( os.path.join(opt.save_checkpoints_dir, opt.exper_name, "D") )

# =====================
# Visualation settings
# =====================
board_train = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.exper_name))
# board_test = SummaryWriter( log_dir = os.path.join(opt.tensorboard_dir, opt.exper_name + "_test") )

# ========
# dataset 
# ========
if(opt.dataset == "folder"):
    transform = transforms.Compose(
        [
            transforms.Resize(opt.imageSize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = dset.ImageFolder(
        root=opt.dataroot,
        transform=transform,
    )
elif(opt.dataset == "mnist"):
    transform = transforms.Compose(
        [
            transforms.Resize(opt.imageSize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = dset.MNIST(
        root=opt.dataroot,
        download=True,
        transform=transform,
        train=True,
    )
elif(opt.dataset == "cifar-10"):
    transform = transforms.Compose(
        [
            transforms.Resize(opt.imageSize, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = dset.CIFAR10(
        root=opt.dataroot,
        download=True,
        transform=transform,
        train=True,
    )
else:
    raise NotImplementedError('dataset %s not implemented' % opt.dataset)

# ===========
# dataloader
# ===========
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
    )

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
if(opt.dataset == "mnist"):
    generator = dcgan.Generator(img_size=opt.imageSize, nz=opt.nz, nc=1, ngf=opt.ngf).to(device)
    discriminator = dcgan.Discriminator(img_size=opt.imageSize, nc=1, ndf=opt.ndf).to(device)
else:
    generator = dcgan.Generator(img_size=opt.imageSize, nz=opt.nz, nc=3, ngf=opt.ngf).to(device)
    discriminator = dcgan.Discriminator(img_size=opt.imageSize, nc=3, ndf=opt.ndf).to(device)

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
z_fixed = Variable(Tensor(np.random.normal(0, 1, (opt.batchSize, opt.nz, 1, 1)))).to(device)

# Initialize BCELoss function
criterion = nn.BCELoss()

fake_images_historys = []

# ======
# train
# ======
iterations = 0
for epoch in range(opt.nepochs):
    for i, (imgs, _) in enumerate(dataloader):
        generator.train()
        discriminator.train()

        iterations += opt.batchSize

        # Adversarial ground truths
        valid = torch.full((imgs.shape[0],), real_label, device=device)
        fake = torch.full((imgs.shape[0],), fake_label, device=device)
        
        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)

        for param in discriminator.parameters():
            param.requires_grad = True

        # ====================
        # Train the discriminator
        # ====================
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz, 1, 1)))).to(device)

        # Generate a batch of images
        with torch.no_grad():
            fake_imgs = generator(z)
            
        # Adversarial loss
        lossD_real = criterion(discriminator(real_imgs), valid)
        lossD_fake = criterion(discriminator(fake_imgs.detach()), fake)
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

        # if i % opt.n_critic == 0:
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz, 1, 1)))).to(device)

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)
        # Adversarial loss
        lossG = criterion(discriminator(gen_imgs), valid)
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
            board_add_image(board_train, 'fake_image', gen_imgs, iterations)
            # Monitor trainnig progresses
            print("epoch={}, iters={}, loss_G={:.5f}, loss_C={:.5f}".format(epoch, iterations, lossG, lossD))
        
    # ============
    # Save images
    # ============
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(z_fixed)
    
    save_image(fake_imgs[0], os.path.join(opt.dir_out, opt.exper_name) + "/fake_image_epoches{}_batch0.png".format(epoch), normalize=True)
    save_image(fake_imgs, os.path.join(opt.dir_out, opt.exper_name) + "/fake_image_epoches{}_batchAll.png".format(epoch), nrow=int(np.sqrt(opt.batchSize)), normalize=True)

    # =====================
    # Save trainnig models
    # =====================
    save_checkpoint_wo_step(generator, device, os.path.join(opt.save_checkpoints_dir, opt.exper_name, "G", 'G_final.pth'))
    save_checkpoint_wo_step(discriminator, device, os.path.join(opt.save_checkpoints_dir, opt.exper_name, "D", 'D_final.pth'))

    # 生成画像のGIF作成(データサイズが増えるため適宜コメントを外す)
    # fake_images_historys.append(fake_imgs[0].transpose(0,1).transpose(1,2).cpu().clone().numpy())
    # save_image_historys_gif(fake_images_historys, os.path.join(opt.dir_out, opt.exper_name) + "/fake_image_epoches{}.gif".format( epoch ))

generator.eval()
score_z = Variable(Tensor(np.random.normal(0, 1, (1024, opt.nz, 1, 1)))).to(device)
score_imgs = generator(score_z)
score_mean, score_std = inception_score(imgs=score_imgs, cuda=cuda, batch_size=32, resize=True, splits=1)
print(score_mean, score_std)