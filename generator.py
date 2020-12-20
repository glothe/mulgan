from torch import nn, optim
from convblock import ConvBlock
from mulsequential import MulSequential



class Generator(MulSequential):
    def __init__(self, nfc: int, nfc_min: int, scale: int, device, opt):
        super().__init__()
        self.scale = scale
        self.device = device
        self.opt = opt

        N = int(nfc)

        self.add_module("head", ConvBlock(
            in_channels=opt.nc_image, 
            out_channels=N, 
            padding=1))
    
        for i in range(opt.num_layers - 2):
            N = int(nfc / 2 ** (i+1))
            block = ConvBlock(
                in_channels=max(2*N, nfc_min),
                out_channels=max(N, nfc_min),
                padding=1,
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

    def step(self, discriminator, image_real,
             noise_fake, previous_fake, 
             noise_reconstruction, previous_reconstruction):

        self.zero_grad()
        self.train()
        
        # Fake generation
        image_fake = self(noise_fake, previous_fake)
        discriminator.eval()
        output = discriminator(image_fake)
        errG_gen = -output.mean()
        errG_gen.backward()  # MEF: that updates discriminator gradients too I think
        discriminator.zero_grad()

        # Reconstruction
        image_reconstructed = self(noise_reconstruction, previous_reconstruction)
        errG_rec = self.opt.alpha * nn.MSELoss()(image_reconstructed, image_real)
        errG_rec.backward()

        # Optimizer
        self.optimizer.step()

        return image_reconstructed.detach()
