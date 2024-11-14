
import torch

class AddNoise(object):
    """Transform to add Gaussian noise to images"""
    def __init__(self, noise_factor=0.0):
        self.noise_factor = noise_factor

    def __call__(self, x):
        if self.noise_factor <= 0.0:
            return x
        noisy = x + self.noise_factor * torch.randn_like(x)
        return torch.clamp(noisy, 0., 1.)