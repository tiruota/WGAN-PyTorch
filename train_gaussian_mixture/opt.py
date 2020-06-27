import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="wgan", help="(wgan , wgan-gp or wgan-div)")
    parser.add_argument('--iterate', type=int, default=5000)
    parser.add_argument('--num_mixture', type=int, default=8)
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--device', default="gpu", help="(CPU or GPU)")
    parser.add_argument('--dir_out', default="./", help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--exper_name', default="WGAN_train", help='Where to store samples and models')
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="path to TensorBoard")
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nc', type=int, default=2, help='input image channels')
    parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--lambda_gp', type=int, default=10, help="Loss weight for gradient penalty")
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--p', type=int, default=6)
    parser.add_argument('--n_display_step', type=int, default=100, help="step of plot to tensorboard")
    parser.add_argument("--n_save_step", type=int, default=5000, help="step of making checkpoints")
    parser.add_argument('--n_critic', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--optimizer', default="rmsprop", help="Whether to use adam (default is rmsprop)")
    opt = parser.parse_args()

    return opt