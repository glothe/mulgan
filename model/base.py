import torch
import os
import numpy as np
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, padding: int = 0, batch_norm=True):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            padding_mode="reflect") # TODO investigate this
        ) 
        if batch_norm:
            self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("leaky_relu", nn.LeakyReLU(0.2, inplace=True))


class BaseModel(nn.Sequential):
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
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in self.trainable_parameters])

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
