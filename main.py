import argparse

import torch

from model.mulgan import MulGAN


parser = argparse.ArgumentParser()

parser.add_argument('--nc_image',type=int, help='image # channels',default=3)
parser.add_argument('--images', nargs='*', type=str, help='Paths of images to use.', default=["balloons.png"])

# Networks hyper parameters:
parser.add_argument('--nfc', type=int, default=32)
parser.add_argument('--nfc_min', type=int, default=32)
parser.add_argument('--num_layers', type=int, help='number of layers', default=5)

# Pyramid parameters:
parser.add_argument('--scale_factor',type=float,help='pyramid scale factor', default=0.75)
parser.add_argument('--min_size', type=int,help='image minimal size at the coarser scale', default=25)
parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

# Optimization hyper parameters
parser.add_argument('--Niter', type=int, default=2000, help='number of epochs to train per scale')
parser.add_argument('--gamma', type=float,help='scheduler gamma',default=0.1)
parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.001')
parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--G_steps', type=int, help='Generator inner steps',default=3)
parser.add_argument('--D_steps', type=int, help='Discriminator inner steps',default=3)
parser.add_argument('--batch_size', type=float, help='fake image batch size',default=1)

opt = parser.parse_args()

model = MulGAN(opt)
model.train()
