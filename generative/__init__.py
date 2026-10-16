from typing import Tuple
import torch.nn as nn
from arch.unet import UNet
from generative.ddpm import DDPM
from generative.flow_matching import FlowMatching

METHODS = {'ddpm': DDPM, 'flow': FlowMatching}


def build_method(name: str, sample_shape: Tuple[int, ...], **unet_kwargs) -> nn.Module:
    """Wrap a fresh U-Net in the requested generative method ('ddpm' or 'flow')."""
    return METHODS[name](UNet(**unet_kwargs), sample_shape)
