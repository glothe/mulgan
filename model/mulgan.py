import os
import math

import torch
from torch import nn, optim
import torchvision
from PIL import Image

import tqdm

from model.generator import Generator
from model.discriminator import Discriminator
from utils import *

from scipy.interpolate import CubicSpline


# Assumption for MulGAN: All images have the same size, and they are somehow coherent
# (frames of a video, different points of view of a scene taken from different angles...)
class MulGAN():
    def __init__(self, opt):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using {self.device}")

        self.opt = opt

        # Sizes
        self.sizes = []
        self.nscales = 0

        # Input images
        self.im_paths = opt.images
        self.nimages = len(self.im_paths)
        self.scaled_images = []

        self.build_pyramid()

        # NNs - the discriminator doesn't need to be stored
        self.generators = []

        # Reconstruction
        self.noise_amplifications = []
        self.rec_input_noises = []
        self.rec_input_images = []        

    def generate_noise(self, size: tuple, batch_size: int = 1):
        noise = torch.randn(batch_size, 3, *size, device=self.device)
        # noise = noise.expand(batch_size, 3, *size)
        return noise

    def train(self):
        """Full end to end training method"""
        opt = self.opt

        nfc_prev = None

        for scale in range(self.nscales):
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
        real_image = self.scaled_images[scale].to(self.device)

        # Input for reconstruction loss
        self.build_reconstruction_input(scale)

        for epoch in tqdm.tqdm(range(opt.Niter), ncols=80):
            fake_input_noise, fake_input_image = self.build_fake_input(scale, batch_size=opt.batch_size)

            # Train generator
            # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L169
            for i in range(opt.G_steps):
                fake_image, rec_image = generator.step(
                    discriminator,
                    real_image,
                    fake_input_noise,
                    fake_input_image,
                    self.rec_input_noises[scale],
                    self.rec_input_images[scale])

            # Train discriminator
            # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L105
            for i in range(opt.D_steps):
                discriminator.step(real_image, fake_image)

            if (epoch + 1) % 100 == 0:
                torchvision.utils.save_image(denormalize_image(fake_image), f"images/{scale}/fake_image.png")
                torchvision.utils.save_image(denormalize_image(rec_image), f"images/{scale}/rec_image.png")

            discriminator.scheduler.step()
            generator.scheduler.step()
    
        discriminator.save()
        generator.save()

    def build_pyramid(self, out_folder: str = "images"):
        """Initializes the attributes self.scaled_images, self.sizes and self.nscales.
        self.sizes is the list of all chosen sizes.
        self.nscales is the number of such scales.
        self.scaled_images is a list of tensors containing the rescaled images,
        indexed first by scale then by image.
        i.e. self.scaled_images[n][m] is the n-th scale of the m_th image."""

        # Get all input images
        paths = self.im_paths
        input_images = []
        for im_path in paths:
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
        for scale, size in enumerate(self.sizes):
            scaled_images = []
            for im_idx, im in enumerate(input_images):
                # TODO: use bicubic interpolation ?
                # Should be better and the additional time cost should not be a problem since it's done only once
                resized_image = torchvision.transforms.functional.resize(im, size)
                resized_image = torchvision.transforms.functional.to_tensor(resized_image)
                resized_image = normalize_image(resized_image[None,:])
                scaled_images.append(resized_image.to(self.device))

                # Save this image
                scale_folder = os.path.join(out_folder, str(scale))
                if not os.path.exists(scale_folder):
                    os.mkdir(scale_folder)
                path = os.path.join(scale_folder, os.path.basename(paths[im_idx]))
                torchvision.utils.save_image(denormalize_image(resized_image), path)
            self.scaled_images.append(torch.cat(scaled_images, dim=0))

    def build_reconstruction_input(self, scale):
        # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L243
        batchsize, nc, nx, ny = self.scaled_images[scale].shape

        if len(self.rec_input_noises) <= scale:
            rec_input_image_path = os.path.join("images", str(scale), f"rec_input_image.pth")
            rec_input_noise_path = os.path.join("images", str(scale), f"rec_input_noise.pth")
            if os.path.exists(rec_input_noise_path):
                # Reconstruction inputs for this scale and this image already exists
                image = torch.load(rec_input_image_path)
                noise = torch.load(rec_input_noise_path)
            else:
                # Generate reconstruction inputs
                if scale == 0:
                    image = torch.zeros(self.nimages, 1, 1, 1, device=self.device)
                    noise = self.generate_noise([nx, ny], self.nimages)
                    # TODO: Find better reconstruction inputs, this is too random
                else:
                    rec_input_image = self.rec_input_images[scale - 1]
                    rec_input_noise = self.rec_input_noises[scale - 1]
            
                    with torch.no_grad():
                        image = self.generators[scale - 1](rec_input_noise, rec_input_image)

                    image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                    noise = torch.zeros_like(image)

                torch.save(image, rec_input_image_path)
                torch.save(noise, rec_input_noise_path)

            # Compute noise amplification
            if scale == 0:
                noise_amplification = 1
            else:
                noise_amplification = torch.sqrt(nn.functional.mse_loss(image, self.scaled_images[scale])).item()

            print(f"Noise amplification for scale {scale}: {noise_amplification:.5f}")
            self.noise_amplifications.append(noise_amplification)
            self.rec_input_images.append(image)
            self.rec_input_noises.append(noise)

    def build_fake_input(self, scale, batch_size=1):
        # https://github.com/tamarott/SinGAN/blob/master/SinGAN/training.py#L224
        batchsize, nc, nx, ny = self.scaled_images[0].shape
        image = torch.zeros(batch_size, 1, nx, ny, device=self.device)   
        noise = self.generate_noise([nx, ny], batch_size) * self.noise_amplifications[0]

        with torch.no_grad():
            for scale, generator in enumerate(self.generators[:scale]):
                image = generator(noise, image)
                bs, nc, nx, ny = self.scaled_images[scale+1].shape
                image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                noise = self.generate_noise([nx, ny], batch_size) * self.noise_amplifications[scale + 1]

        return noise, image

    def gen_only0(self, noise):
        batch_size = noise.shape[0]
        image = torch.zeros(batch_size, 1, 1, 1, device=self.device)
        with torch.no_grad():
            for scale, generator in enumerate(self.generators):
                _, _, nx, ny = self.scaled_images[scale].shape
                image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                if scale == 0:
                    image = generator(noise, image)
                else:
                    image = generator(torch.zeros_like(image), image)    
        return image

    def gen(self, noises, batch_size=1):
        assert len(noises) == self.nscales
        image = torch.zeros(batch_size, 1, 1, 1, device=self.device)
        with torch.no_grad():
            for scale, generator in enumerate(self.generators):
                _, _, nx, ny = self.scaled_images[scale].shape
                image = torchvision.transforms.functional.resize(image, size=(nx, ny))
                image = generator(noises[scale], image)
        return image

    def gen_random_image(self, scale=None, batch_size=1, scales_to_sample="only0"):
        assert scales_to_sample in ["only0", "all"]
        if scale is None:
            scale = len(self.sizes) - 1
        noises = []
        for scale in range(self.nscales):
            _, _, nx, ny = self.scaled_images[scale].shape
            if scales_to_sample == "only0" or scale == 0:
                noises.append(self.generate_noise([nx, ny], batch_size) * self.noise_amplifications[scale])
            else:
                noises.append(torch.zeros((batch_size, nx, ny), device=self.device))
        return self.gen(noises, batch_size)

    def markov_walk(self, init_pos="rec", step_std=1, n_images=10, scales_to_sample="only0"):
        assert init_pos in ["rec", "zeros", "random"] or type(init_pos) is list
        assert scales_to_sample in ["only0", "all"]
        if init_pos == "rec":
            init_pos = [rin[0] for rin in self.rec_input_noises]
        elif init_pos == "zeros":
            init_pos = []
            for i in range(self.nscales):
                _, _, nx, ny = self.scaled_images[i].shape
                init_pos.append(torch.zeros((3, nx, ny), device=self.device))
        elif init_pos == "random":
            init_pos = []
            for i in range(self.nscales):
                _, _, nx, ny = self.scaled_images[i].shape
                init_pos.append(self.generate_noise([nx, ny], 1) * self.noise_amplifications[i])
        noises = []
        for scale in range(self.nscales):
            _, _, nx, ny = self.scaled_images[scale].shape
            noise = torch.zeros((n_images, 3, nx, ny), device=self.device)
            if scales_to_sample == "all" or scale == 0:
                noise[0] = init_pos[scale]
                for i in range(1, n_images):
                    # TODO: Find better way of sampling along the distribution we are supposed to sample
                    # That is, centered gaussian with std self.noise_amplifications[scale]
                    noise[i] = noise[i - 1] + torch.normal(mean=torch.zeros_like(noise[i - 1]), std=step_std) * self.noise_amplifications[scale]
            noises.append(noise)  
        return self.gen(noises, n_images)

    def linear_interpolate(self, freq=2, batch_size=8):
        # Just shift the input noises a step towards the left so that
        # the interpolations we want are between this_frame_in[i] and next_frame_in[i]
        this_frame_in = self.rec_input_noises[0][:-1]
        next_frame_in = self.rec_input_noises[0][1:]
        interpolated_noises = torch.zeros((freq, *this_frame_in.shape), device=self.device)
        for i, t in enumerate(torch.linspace(0, 1, freq)):
            interpolated_noises[i] = (1 - t) * this_frame_in + t * next_frame_in
        # interpolated_noises is now a tensor of all the frames with the first dimension corresponding to the t
        # We need to interleave the first two dimensions to get a batch of the right input noises
        interpolated_noises.transpose_(0, 1)
        interpolated_noises = interpolated_noises.reshape(-1, *interpolated_noises.shape[2:])
        out = torch.zeros((self.nimages * freq, *self.scaled_images[-1].shape[1:]))
        n_batches = self.nimages * freq // batch_size
        for batch_idx in range(n_batches):
            out[batch_idx * batch_size : (batch_idx + 1) * batch_size] = self.gen_only0(interpolated_noises[batch_idx * batch_size : (batch_idx + 1) * batch_size])
        if n_batches * batch_size < self.nimages * freq:
            out[n_batches * batch_size:] = self.gen_only0(interpolated_noises[n_batches * batch_size:])
        return out
        return self.gen_only0(interpolated_noises)

    def cubic_interpolate(self, freq=2, batch_size=8):
        xs = range(self.nimages)
        ys = self.rec_input_noises[0]
        interpolator = CubicSpline(xs, ys.cpu().numpy())
        new_noises = interpolator(torch.linspace(0, self.nimages - 1, self.nimages * freq))
        new_t_noises = torch.tensor(new_noises, device=self.device, dtype=torch.float32)
        out = torch.zeros((self.nimages * freq, *self.scaled_images[-1].shape[1:]))
        n_batches = self.nimages * freq // batch_size
        for batch_idx in range(n_batches):
            out[batch_idx * batch_size : (batch_idx + 1) * batch_size] = self.gen_only0(new_t_noises[batch_idx * batch_size : (batch_idx + 1) * batch_size])
        if n_batches * batch_size < self.nimages * freq:
            out[n_batches * batch_size:] = self.gen_only0(new_t_noises[n_batches * batch_size:])
        return out

