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
