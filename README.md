
# Wasserstein GAN - Pytorch
[ [paper](https://arxiv.org/abs/1701.07875) ] [ [著者のリポジトリ](https://github.com/martinarjovsky/WassersteinGAN) ]

- WGANをPytorchで実装しました．比較対象としてDCGANも実装しました．
    - [Notebook](https://dev.usagee.jp/t183350/wgan-pytorch/blob/master/wgan.ipynb)
- モード崩壊と回避を可視化しました．
    - [Notebook](https://dev.usagee.jp/t183350/wgan-pytorch/blob/master/train_gaussian_mixture/wgan_gaussian_mixture.ipynb)


参考コード
- https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_WGAN_PyTorch
- https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py


# Examples

- MNIST
```bash
python wgan.py --dataset mnist --batchSize 25 --nepochs 200
```

- CIFAR-10
```bash
python wgan.py --dataset cifar-10 --batchSize 25 --nepochs 200
```

- 自作データセット

DataLoaderの仕様上，自作データセットフォルダにサブフォルダーを作成し，その中にデータを入れてください．

(hoge_dataset/train/hoge1.png のように)

```bash
python wgan.py --dataset folder --dataroot [Your Dataset Folder] --batchSize 64 --nepochs 200
```

- Gaussian Mixture

```bash
cd train_gaussian_mixture
python train_wgan.py --batchSize 64 --nepochs 200
```

# Results

## MNIST

![mnist](https://imgur.com/SRFHA2y.gif)

## CIFAR-10

![cifar10_random](https://imgur.com/eGhCYHd.gif)

## Emoji

データセット：https://www.joypixels.com/

![emoji_random](https://imgur.com/yKLUyGd.gif)

## Gaussian Mixture

- scatter

|Ground-truth|DCGAN|WGAN|
|:---:|:---:|:---:|
|<img src="https://imgur.com/eZ65BmP.png" width="250">|<img src="https://imgur.com/kF5XSug.gif" width="250">|<img src="https://imgur.com/NKNetmt.gif" width="250">|

- KDE

|Ground-truth|DCGAN|WGAN|
|:---:|:---:|:---:|
|<img src="https://imgur.com/8Ld9KJY.png" width="250">|<img src="https://imgur.com/mFeEDEy.gif" width="250">|<img src="https://imgur.com/7mzkZ1l.gif" width="250">|