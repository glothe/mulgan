import os
import math

import torch
torch.autograd.set_detect_anomaly(True)

from torch import nn, optim
import torchvision
from PIL import Image

import tqdm


def normalize_image(image):
    return (image * 2 - 1).clamp(-1, 1)

def denormalize_image(image):
    return (image * .5 + .5).clamp(-1, 1)

def generate_sizes(max_size=250, min_size=25, scale_factor=0.75):
    max_size = 250
    size_factor = 1
    size = max_size
    sizes = [max_size]

    while size > 25:
        size_factor *= 0.75
        size = int(size_factor * max_size)

        sizes.append(size)

    print(sizes)
    return sizes[::-1]

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, padding: int = 0, batch_norm=True):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding))
        if batch_norm:
            self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("leaky_relu", nn.LeakyReLU(0.2, inplace=True))


class Sequential(nn.Sequential):
    """ Base class for Generator & Discriminator. """
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def init_parameters(self, module):
        classname = module.__class__.__name__
        if classname.find('Conv2d') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def parameter_count(self):
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
        return sum([np.prod(p.size()) for p in model_parameters])

    def save(self):
        torch.save(self.state_dict(), f"images/{self.scale}/{self.__class__.__name__}.pth")

    def load(self, scale: int = None, verbose: bool = True):
        if scale is None:
            scale = self.scale

        path = f"images/{scale}/{self.__class__.__name__}.pth"
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))

            if verbose:
                print(f"\t{self.__class__.__name__} recovered from previous training.")

            return True

        return False


class Discriminator(Sequential):
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
        ))

        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[1600], gamma=opt.gamma)

        self.to(device)
        self.apply(self.init_parameters)

    def gradient_penalty(self, real_image, fake_image):
        # *Memory leak somewhere in this function*
        print(torch.cuda.memory_allocated(0) / 2**20)
        alpha = torch.rand(1, 1)
        alpha.expand(real_image.size())
        alpha = alpha.to(self.device)

        interpolation = alpha * real_image + (1 - alpha) * fake_image
        interpolation.requires_grad = True

        output = self(interpolation).mean()  # original paper was .sum()

        print(torch.cuda.memory_allocated(0) / 2**20)

        gradient = torch.autograd.grad(output, interpolation, 
            create_graph=True, only_inputs=True)[0]

        print(torch.cuda.memory_allocated(0) / 2**20)

        gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_grad
        gradient_penalty.backward()

        # This does not solve anything
        del interpolation
        del output
        del gradient
        del gradient_penalty
        torch.cuda.empty_cache()

        print(torch.cuda.memory_allocated(0) / 2**20)
        print()

    def update(self, image_real, image_fake):
        for i in range(self.opt.D_steps):
            self.zero_grad()

            # Train with real image
            output = self(image_real)
            errD_real = -output.mean()
            errD_real.backward()

            # Train with fake
            output = self(image_fake)
            errD_fake = output.mean()
            errD_fake.backward()

            self.gradient_penalty(image_real, image_fake)

            # Optimizer
            self.optimizer.step()


class Generator(Sequential):
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
        ))
        self.add_module("tail_tanh", nn.Tanh())

        # Optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[1600], gamma=opt.gamma)

        self.to(device)
        self.apply(self.init_parameters)

    def forward(self, noise, previous_image = 0):
        generated = super().forward(noise)
        return generated + previous_image

    def update(self, discriminator, image_real, image_fake, noise_reconstruction, previous_reconstruction):
        # opt.G_steps > 1 causes gradient issues
        # Using G_steps > 1 isn't really useful anyway as the generator
        # is only applied once anyway.

        self.zero_grad()

        # Generation loss
        output = discriminator(image_fake) 
        errG_gen = -output.mean()
        errG_gen.backward()

        # Reconstruction loss
        image_reconstructed = self(noise_reconstruction, previous_reconstruction)
        errG_rec = self.opt.alpha * nn.MSELoss()(image_reconstructed, image_real)
        errG_rec.backward()

        # Optimizer
        self.optimizer.step()

        return image_reconstructed


class SinGAN():
    def __init__(self, opt):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using {self.device}")

        self.opt = opt

        self.build_pyramid()
        self.generators = []

        # Reconstruction
        self.noise_amplifications = []
        self.noises_reconstruction = []
        self.images_reconstruction = []        

    def generate_noise(self, size):
        noise = torch.randn(1, 1, *size, device=self.device)
        noise = noise.expand(1, 3, *size)
        return noise

    def train(self):
        opt = self.opt

        nfc_prev = None

        for scale in range(len(self.images)):
            print(f"Scale: {scale}")
            nfc = min(opt.nfc * pow(2, math.floor(scale / 4)), 128)
            nfc_min = min(opt.nfc_min * pow(2, math.floor(scale / 4)), 128)

            # Discriminator set up
            discriminator = Discriminator(nfc, nfc_min, scale, self.device, opt)
            discriminator_loaded = discriminator.load()

            # Generator set up
            generator = Generator(nfc, nfc_min, scale, self.device, opt)
            generator_loaded = generator.load()
            self.generators.append(generator)

            # TODO: if the number of feature doesnt' increase, use the previous layer
            # as pre-trained model

            if discriminator_loaded and generator_loaded:
                self.build_reconstruction_input(scale)
                print(f"Skipping scale {scale}.")
            else:
                # Attempt to load the previous layer as pre-training
                if nfc_prev == nfc and scale > 0:
                    if not discriminator_loaded:
                        discriminator.load(scale=scale-1, verbose=False)
                    if not generator_loaded:
                        generator.load(scale=scale-1, verbose=False)
            
                self.train_scale(scale, discriminator, generator)

            generator.freeze()
            nfc_prev = nfc

    def train_scale(self, scale, discriminator, generator):
        opt = self.opt
        image_real = self.images[scale].to(self.device)

        # Input for reconstruction loss
        noise_amplification, noise_reconstruction, previous_reconstruction = self.build_reconstruction_input(scale)

        for epoch in tqdm.tqdm(range(opt.Niter), ncols=80):
            noise_fake, previous_fake = self.build_fake_input(scale)
            image_fake = generator(noise_fake, previous_fake)
            
            discriminator.update(
                image_real, 
                image_fake.detach())

            image_reconstructed = generator.update(
                discriminator, 
                image_real, 
                image_fake, 
                noise_reconstruction, 
                previous_reconstruction)

            # if (epoch + 1) % 100 == 0:
            #     torchvision.utils.save_image(denormalize_image(image_fake), f"images/{scale}/image_fake.png")
            #     torchvision.utils.save_image(denormalize_image(image_reconstructed), f"images/{scale}/image_reconstructed.png")

            # discriminator.scheduler.step()
            # generator.scheduler.step()
    
        discriminator.save()
        generator.save()

    def build_pyramid(self, path="balloons.png", out_folder="images"):
        with open(path, "rb") as f:
            with Image.open(f) as image:
                image = image.convert("RGB")

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        sizes = generate_sizes(
            max_size=min(self.opt.max_size, *image.size), 
            min_size=self.opt.min_size)
        self.images = []

        for scale, size in enumerate(sizes):
            resized_image = torchvision.transforms.functional.resize(image, size)
            resized_image = torchvision.transforms.functional.to_tensor(resized_image)
            resized_image = normalize_image(resized_image[None,:])
            self.images.append(resized_image.to(self.device))

            scale_folder = os.path.join(out_folder, str(scale))
            if not os.path.exists(scale_folder):
                os.mkdir(scale_folder)
            path = os.path.join(scale_folder, os.path.basename(path))
            torchvision.utils.save_image(denormalize_image(resized_image), path)

    def build_reconstruction_input(self, scale):
        _, nc, nx, ny = self.images[scale].shape

        if len(self.noises_reconstruction) <= scale:
            if os.path.exists(f"images/{scale}/noise_amplification.pth"):
                # Reconstruction inputs for this scale already exists
                noise_amplification = torch.load(f"images/{scale}/noise_amplification.pth")
                image = torch.load(f"images/{scale}/image_reconstruction.pth")
                noise = torch.load(f"images/{scale}/noise_reconstruction.pth")

            else:
                # Generate reconstruction inputs
                if scale == 0:
                    image = torch.zeros(1, 1, 1, 1, device=self.device)
                    noise = self.generate_noise([nx, ny])
                    noise_amplification = 1

                else:
                    image_previous = self.images_reconstruction[scale-1]
                    noise_previous = self.noises_reconstruction[scale-1]
                    with torch.no_grad():
                        image = self.generators[scale-1](noise_previous, image_previous)
                    image = torchvision.transforms.functional.resize(image, size=(nx, ny))  
                    noise_amplification = torch.sqrt(nn.MSELoss()(image, self.images[scale]))
                    noise = torch.zeros([1, 3, nx, ny], device=self.device) * noise_amplification + image

                torch.save(noise_amplification, f"images/{scale}/noise_amplification.pth")
                torch.save(image, f"images/{scale}/image_reconstruction.pth")
                torch.save(noise, f"images/{scale}/noise_reconstruction.pth")

            print(f"Noise amplification for scale {scale}: {noise_amplification:.5f}")
    
            self.noise_amplifications.append(noise_amplification)
            self.noises_reconstruction.append(noise)
            self.images_reconstruction.append(image)
            
        return \
            self.noise_amplifications[scale], \
            self.noises_reconstruction[scale], \
            self.images_reconstruction[scale]

    def build_fake_input(self, target_scale):
        _, nc, nx, ny = self.images[0].shape
        previous_fake = torch.zeros(1, 1, 1, 1, device=self.device)   
        noise_fake = self.generate_noise([nx, ny]) # * noise_amplication[0] + previous_fake     

        with torch.no_grad():
            for scale, generator in enumerate(self.generators[:target_scale]):
                previous_fake = generator(noise_fake, previous_fake)

                _, nc, nx, ny = self.images[scale+1].shape
                previous_fake = torchvision.transforms.functional.resize(previous_fake, size=(nx, ny))
                noise_fake = self.generate_noise([nx, ny]) * self.noise_amplifications[scale+1] + previous_fake

        return noise_fake, previous_fake



