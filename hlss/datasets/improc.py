"""Image processing functions designed to work with OpenSRH and TCGA datasets.

Copyright (c) 2024 Mohamed Bin Zayed University of Artificial Intelligence. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Optional, List, Tuple, Dict
from functools import partial
from PIL import Image
import random
import tifffile
import numpy as np
import h5py
import torch
from torch.nn import ModuleList
from torchvision.transforms import (
    Normalize, RandomApply, Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, RandAugment, RandomErasing, RandomAutocontrast, Grayscale,
    RandomSolarize, ColorJitter, RandomAdjustSharpness, GaussianBlur,
    RandomAffine, RandomResizedCrop, CenterCrop)
from torchvision.transforms import Compose, Resize,ToPILImage, CenterCrop, ToTensor, Normalize, RandomResizedCrop

class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image

    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.

    """

    def __init__(self, subtracted_base: float = 5000 / 65536.0):
        super().__init__()
        self.subtracted_base = subtracted_base

    def __call__(self, two_channel_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            two_channel_image: a 2 channel np array in the shape H * W * 2
            subtracted_base: an integer to be added to (CH3 - CH2)

        Returns:
            A 3 channel np array in the shape H * W * 3
        """
        ch2 = two_channel_image[0, :, :]
        ch3 = two_channel_image[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return torch.stack((ch1, ch2, ch3), dim=0)


class MinMaxChop(torch.nn.Module):
    """Clamps the images to float (0,1) range."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(self.min_, self.max_)


class GaussianNoise(torch.nn.Module):
    "