import os
import math
import torch
from torch import nn, optim
import torchvision
from PIL import Image
import tqdm
from generator import Generator
from discriminator import Discriminator
from utils import *


# TODO: Use tensors instead of lists when possible
# TODO: Use batches to compute multiple images
# TODO: Put tensors on GPU whenever possible, but bring them back to cpu when unused as to not saturate the VRAM
# TODO: Use model.eval() or model.train() before each use of a nn
# Assumption for MulGAN: All images have the same size, and they are somehow coherent
# (frames of a video, different points of view of a scene taken from different angles...)
class MulGAN():
    def __init__(self, opt):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using {self.device}")

        self.opt = opt

        self.nscales = 0
        self.sizes = []
        self.scaled_images = []
        self.im_paths = ["balloons.png"]
        self.build_pyramids()
        self.generators = []

        # Reconstruction
        self.noise_amplifications = []
        self.rec_input_noises = []
        self.rec_input_images = []        

    def generate_noise(self, size, batch_size=1):
        noise = torch.randn(batch_size, 1, *size, device=self.device)
        noise = noise.expand(batch_size, 3, *size)
        return noise

    def train(self):
        """Full end to end training method"""
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

            # Load the saved models
            if discriminator_loaded and generator_loaded:
                self.build_reconstruction_input(scale)
                print(f"Skipping scale {scale}.")
            else:
                # No saved model found
                # Attempt to load the previous layer as pre-training
                if nfc_prev == nfc and scale > 0:
                    if not discriminator_loaded:
                        discriminator.load(scale=scale-1, verbose=False)
                    if not generator_loaded:
                        generator.load(scale=scale-1, verbose=False)
                # In any case, train the model
                self.train_scale(scale, discriminator, generator)

            generator.freeze()
            nfc_prev = nfc

    def train_scale(self, scale, discriminator, generator):
        """Train the discriminator and generator for a single scale"""
        opt = self.opt
        images_real = []
        for im in self.scaled_images[scale]:
            images_real.append(im.to(self.device))

        # Input for reconstruction loss
        self.build_reconstruction_input(scale)

        for epoch in tqdm.tqdm(range(opt.Niter), ncols=80):
            noise_fake, previous_fake = self.build_fake_input(scale, batch_size=opt.batch_size)

            with torch.no_grad():
                image_fake = generator(noise_fake, previous_fake)
            
            # Train discriminator
            # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L105
            for i in range(opt.D_steps):
                for im_r, im_f in zip(images_real, images_fake):
                    discriminator.step(
                        im_r,
                        im_f)

            # Train generator
            # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L169
            for i in range(opt.G_steps):
                image_reconstructed = generator.step(
                    discriminator,
                    image_real,
                    noise_fake,
                    previous_fake,
                    noise_reconstruction,
                    previous_reconstruction)

            if (epoch + 1) % 100 == 0:
                torchvision.utils.save_image(denormalize_image(image_fake), f"images/{scale}/image_fake.png")
                torchvision.utils.save_image(denormalize_image(image_reconstructed), f"images/{scale}/image_reconstructed.png")

            discriminator.scheduler.step()
            generator.scheduler.step()
    
        discriminator.save()
        generator.save()

    def build_pyramids(self, out_folder="images"):
        """Initializes the attributes self.images, self.sizes and self.nscales.
        self.sizes is the list of all chosen sizes.
        self.nscales is the number of such scales.
        self.scaled_images is a list of list containing the rescaled images,
        indexed first by scale then by image.
        i.e. self.scaled_images[n][m] is the n-th scale of the m_th image."""
        paths = self.im_paths
        # Get all input images
        input_images = []
        for im_idx, im_path in enumerate(paths):
            with open(im_path, "rb") as f:
                with Image.open(f) as image:
                    input_images.append(image.convert("RGB"))
        # Create image folder
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        # Choose sizes
        self.sizes = generate_sizes(
            max_size=min(self.opt.max_size, *(min(*image.size) for image in input_images)),
            min_size=self.opt.min_size)
        self.nscales = len(self.sizes)
        # Build all resized images
        self.scaled_images = []
        for scale, size in enumerate(sizes):
            self.scaled_images.append([])
            for im_idx, im in enumerate(input_images):
                # TODO: use bicubic interpolation ?
                # Should be better and the additional time cost should not be a problem since it's done only once
                resized_image = torchvision.transforms.functional.resize(im, size)
                resized_image = torchvision.transforms.functional.to_tensor(resized_image)
                resized_image = normalize_image(resized_image[None,:])
                self.images[-1].append(resized_image.to(self.device))

                scale_folder = os.path.join(out_folder, str(scale))
                if not os.path.exists(scale_folder):
                    os.mkdir(scale_folder)
                path = os.path.join(scale_folder, os.path.basename(paths[im_idx]))
                torchvision.utils.save_image(denormalize_image(resized_image), path)

    def build_reconstruction_input(self, scale):
        # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L243
        _, nc, nx, ny = self.images[scale][0].shape

        if len(self.rec_input_noises) <= scale:
            self.rec_input_images.append([])
            self.rec_input_noises.append([])
            mse_losses = torch.zeros(len(self.im_paths))
            for im_idx, im_path in enumerate(self.im_paths):
                rec_input_image_path = os.path.join("images", str(scale), f"rec_input_image_{im_path}.pth")
                rec_input_noise_path = os.path.join("images", str(scale), f"rec_input_noise_{im_path}.pth")
                if os.path.exists(rec_input_noise_path):
                    # Reconstruction inputs for this scale and this image already exists
                    image = torch.load(rec_input_image_path)
                    noise = torch.load(rec_input_noise_path)
                else:
                    # Generate reconstruction inputs
                    if scale == 0:
                        image = torch.zeros(1, 1, 1, 1, device=self.device)
                        noise = self.generate_noise([nx, ny])  # TODO: Find better reconstruction inputs, this is too random
                    else:
                        image_previous = self.images_reconstruction[scale-1]
                        noise_previous = self.noises_reconstruction[scale-1]
                        with torch.no_grad():
                            self.generators[scale-1].eval()
                            image = self.generators[scale-1](noise_previous, image_previous)
                        image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                        noise = torch.zeros_like(image)

                    torch.save(image, rec_input_image_path)
                    torch.save(noise, rec_input_noise_path)

                if scale != 0:
                    mse_losses[im_idx] = nn.MSELoss()(image, self.images[scale])

                self.rec_input_noises[scale].append(noise)
                self.rec_input_images[scale].append(image)
            
            # Compute noise amplification
            if scale == 0:
                noise_amplification = 1
            else:
                noise_amplification = torch.sqrt(torch.mean(mse_losses)).item()
            print(f"Noise amplification for scale {scale}: {noise_amplification:.5f}")
            self.noise_amplifications.append(noise_amplification)

    def build_fake_input(self, scale, batch_size=1):
        # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L224
        _, nc, nx, ny = self.images[0].shape
        image = torch.zeros(batch_size, 1, nx, ny, device=self.device)   
        noise = self.generate_noise([nx, ny], batch_size) # * noise_amplication[0] + image     

        with torch.no_grad():
            for scale, generator in enumerate(self.generators[:scale]):
                image = generator(noise, image)

                bs, nc, nx, ny = self.images[scale+1].shape
                image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                noise = self.generate_noise([nx, ny], batch_size) * self.noise_amplifications[scale+1] + image

        return noise, image



