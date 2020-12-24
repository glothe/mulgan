import torch
from torch import nn, optim

from model.base import BaseModel, ConvBlock


class Discriminator(BaseModel):
    def __init__(self, nfc: int, nfc_min: int, scale: int, device, opt):
        super().__init__()
        self.scale = scale
        self.device = device
        self.opt = opt

        N = int(nfc)

        self.add_module("head", ConvBlock(
            in_channels=opt.nc_image, 
            out_channels=N, 
            batch_norm=False))
    
        for i in range(opt.num_layers - 2):
            N = int(nfc / 2 ** (i+1))
            block = ConvBlock(
                in_channels=max(2*N, nfc_min),
                out_channels=max(N, nfc_min),
                batch_norm=False
            )
            self.add_module(f"block{i+1:d}", block)
        
        self.add_module("tail", nn.Conv2d(
            in_channels=max(N, nfc_min),
            out_channels=1,
            kernel_size=3,
            padding_mode="reflect" # TODO investigate this
        ))

        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[1600], gamma=opt.gamma)

        self.to(device)
        self.apply(self.init_parameters)

    def gradient_penalty(self, real_image, fake_image):
        """Compute the parameters gradients for the WGAN-GP loss"""
        alpha = torch.rand(1, device=self.device)
        interpolation = alpha * real_image + (1 - alpha) * fake_image
        interpolation.requires_grad = True

        output = self(interpolation)
        gradient = torch.autograd.grad(output.sum(), interpolation,
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def gradient_penalty_zero(self, real_image, real_output):
        batch_size = real_image.shape[0]

        gradient = torch.autograd.grad(
            outputs=real_output.sum(), inputs=real_image,
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = gradient.pow(2).view(batch_size, -1).sum(1)
        return gradient_penalty

    def step(self, real_image, fake_image):
        self.zero_grad()
        self.train()

        # https://github.com/FriedRonaldo/SinGAN/blob/master/code/train.py#L108
        if self.opt.gan_type == "wgan-gp":
            # Train with real image
            real_output = self(real_image)
            loss_real = -real_output.mean()
            loss_real.backward()

            # Train with fake
            fake_output = self(fake_image)
            loss_fake = fake_output.mean()
            loss_fake.backward()

            # WGAN-GP loss
            loss_gp = 0.1 * self.gradient_penalty(real_image, fake_image)
            loss_gp.backward()

        elif self.opt.gan_type == "zero-gp":
            # Train with fake
            fake_output = self(fake_image)
            zeros = torch.zeros_like(fake_output, device=self.device)
            loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_output, zeros, reduction='none').mean()

            # Train with real image
            real_image.requires_grad = True
            real_output = self(real_image)
            ones = torch.ones_like(real_output, device=self.device)
            loss_real = nn.functional.binary_cross_entropy_with_logits(real_output, ones, reduction='none').mean()

            # Zero-GP loss
            loss_gp = self.gradient_penalty_zero(real_image, torch.mean(real_output, (2, 3))).mean()
            print("\nlosses:\n", loss_real, loss_fake, loss_gp)
            loss = loss_fake + loss_real + 10 * loss_gp
            loss.backward()

            real_image.requires_grad = False
            real_image.grad = None

        # Optimizer
        self.optimizer.step()
