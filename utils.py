# -*- coding:utf-8 -*-
import os
from PIL import Image
import imageio

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from scipy.stats import entropy
from torchvision.models.inception import inception_v3

#====================================================
# モデルの保存＆読み込み関連
#====================================================
def save_checkpoint(model, device, save_path, step):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(
        {
            'step': step,
            'model_state_dict': model.cpu().state_dict(),
        }, save_path
    )
    model.to(device)
    return

def save_checkpoint_wo_step(model, device, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)
    return

def load_checkpoint(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint['step']
    model.to(device)
    return model, step

def load_checkpoint_wo_step(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    return model

#====================================================
# 画像の保存関連
#====================================================
def save_image_historys_gif( images_historys, file_name = "images_historys.gif" ):
    """
    画像履歴を gif ファイルで保存
    """
    imageio.mimsave( file_name, images_historys )
    return

#====================================================
# TensorBoard への出力関連
#====================================================
def tensor_for_board(img_tensor):
    # map into [0,1]
    tensor = (img_tensor.clone()+1) * 0.5
    tensor.cpu().clamp(0,1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1,3,1,1)

    return tensor

def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)
    
    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

    return canvas

def board_add_image(board, tag_name, img_tensor, step_count, n_max_images = 32):
    tensor = tensor_for_board(img_tensor)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)

def board_add_images(board, tag_name, img_tensors_list, step_count, n_max_images = 32):
    tensor = tensor_list_for_board(img_tensors_list)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
            
        Image.fromarray(array).save(os.path.join(save_dir, img_name))

        
#=================
# inception_score
#=================
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)