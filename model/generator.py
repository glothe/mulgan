import torch
from torch import nn, optim

from model.base import BaseModel, ConvBlock


class Generator(BaseModel):
    def __init__(self, nfc: int, nfc_min: int, scale: int, device, opt):
        super().__init__()
        self.scale = scale
        self.device = device
        self.opt = opt

        N = int(nfc)

        self.add_module("head", ConvBlock(
            in_channels=opt.nc_image, 
            out_channels=N, 
            padding=1,
            batch_norm=False))
    
        for i in range(opt.num_layers - 2):
            N = int(nfc / 2 ** (i+1))
            block = ConvBlock(
                in_channels=max(2*N, nfc_min),
                out_channels=max(N, nfc_min),
                padding=1,
                batch_norm=False
            )
            self.add_module(f"block{i+1:d}", block)
        
        self.add_module("tail", nn.Conv2d(
            in_channels=max(N, nfc_min),
            out_channels=opt.nc_image,
            kernel_size=3,
            padding=1,
            padding_mode="reflect" # TODO investigate this
        ))
        self.add_module("tail_tanh", nn.Tanh())

        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[1600], gamma=opt.gamma)

        self.to(device)
        self.apply(self.init_parameters)

    def forward(self, noise, previous_image = 0):
        generated = super().forward(noise + previous_image)
        return generated + previous_image

    def step(self, discriminator, real_image,
             fake_input_noise, fake_input_image, 
             rec_input_noise, rec_input_image):
        self.zero_grad()
        self.train()
        
        # https://github.com/FriedRonaldo/SinGAN/blob/master/code/train.py#L89
        if self.opt.gan_type == "wgan-gp":
            # Generation loss
            fake_image = self(fake_input_noise, fake_input_image)
            fake_output = discriminator(fake_image)
            loss_gen = -fake_output.mean()
            loss_gen.backward()

            # Reconstruction
            rec_image = self(rec_input_noise, rec_input_image)
            loss_rec = 10 * nn.functional.mse_loss(rec_image, real_image)
            loss_rec.backward()
        
        elif self.opt.gan_type == "zero-gp":
            # Generation loss
            fake_image = self(fake_input_noise, fake_input_image)
            fake_output = discriminator(fake_image)
            ones = torch.ones_like(fake_output, device=self.device)
            loss_gen = nn.functional.binary_cross_entropy_with_logits(fake_output, ones, reduction='none').mean()
            loss_gen.backward()

            # Reconstruction
            rec_image = self(rec_input_noise, rec_input_image)
            loss_rec = 100 * nn.functional.mse_loss(rec_image, real_image)
            loss_rec.backward()

        # Optimizer
        self.optimizer.step()

        return fake_image.detach(), rec_image.detach()
