import torch
from torch import nn, optim
from convblock import ConvBlock
from mulsequential import MulSequential


class Discriminator(MulSequential):
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
        gradient = torch.autograd.grad(output, interpolation, 
            grad_outputs=torch.ones(size=output.size(), device=self.device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_grad
        gradient_penalty.backward()

    def step(self, image_real, image_fake):
        self.zero_grad()
        self.train()

        # Train with real image
        output = self(image_real)
        errD_real = -output.mean()
        errD_real.backward()

        # Train with fake
        output = self(image_fake)
        errD_fake = output.mean()
        errD_fake.backward()

        # WGAN-GP loss
        self.gradient_penalty(image_real, image_fake)

        # Optimizer
        self.optimizer.step()
