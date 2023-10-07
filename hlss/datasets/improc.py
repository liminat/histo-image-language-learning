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
    """Adds guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def __call__(self, tensor):

        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy


# def process_read_im(imp: str) -> torch.Tensor:
#     """Read in two channel image

#     Args:
#         imp: a string that is the path to the tiff image

#     Returns:
#         A 2 channel torch Tensor in the shape 2 * H * W
#     """
#     # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133
#     return torch.from_numpy(tifffile.imread(imp).astype(
#         np.float32)).contiguous()

def process_read_im(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """
    # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133
    with tifffile.TiffFile(imp) as tif:
        return torch.from_numpy(tif.asarray().astype(np.float32)).contiguous()


def read_h5_patches(imp: str) -> torch.Tensor:
    """Read image patches from an HDF5 file and return a tensor.

    Args:
        imp (str): Path to the HDF5 file containing image patches.

    Returns:
        torch.Tensor: A tensor containing image patches of shape (n, 300, 300, 3).
    """

    with h5py.File(imp, "r") as h5_file:
        if "imgs" in h5_file:
            dataset = h5_file["imgs"]
            tensor = torch.from_numpy(dataset[:].astype(np.float32)).contiguous()
            
            return tensor
        else:
            print(f'no imgs key')
            raise KeyError("'imgs' key not found in the HDF5 file.")

def read_one_patch(imp: str , idx : int) -> torch.Tensor:
    """Read image patches from an HDF5 file and return a tensor representing one patch.

    Args:
        imp (str): Path to the HDF5 file containing image patches.
        idx (int) : index of the patch number to be read (either 0 or 1)

    Returns:
        torch.Tensor: A tensor containing image patches of shape (n, 300, 300, 3).
    """

    with h5py.File(imp, "r") as h5_file:
        if "imgs" in h5_file:
            im_id = np.random.permutation(np.arange(h5_file["imgs"].shape[0]))
            curr_idx = idx % len(im_id)
            img = h5_file["imgs"][im_id[curr_idx]]
            tensor = torch.from_numpy(img[:].astype(np.float32)).contiguous()
            
            return tensor, im_id[curr_idx]
        else:
            print(f'no imgs key')
            raise KeyError("'imgs' key not found in the HDF5 file.")


def read_400_h5_patches(imp: str) -> torch.Tensor:
    """Read image patches from an HDF5 file and return a tensor representing 400 patches.

    Args:
        imp (str): Path to the HDF5 file containing image patches.

    Returns:
        torch.Tensor: A tensor containing image patches of shape (n, 300, 300, 3).
    """
    tensor_list = []
    idx_list = []
    patches_per_slide = 20

    with h5py.File(imp, "r") as h5_file:
        if "imgs" in h5_file:
            im_id = np.random.permutation(np.arange(h5_file["imgs"].shape[0]))

            for idx in range (patches_per_slide):
                curr_idx = idx % len(im_id)
                img = h5_file["imgs"][im_id[curr_idx]]
                tensor = torch.from_numpy(img[:].astype(np.float32)).contiguous()
                tensor_list.append(tensor)
                idx_list.append(im_id[curr_idx])

            # print(f'tensor list {len(tensor_list)}')
            # print(f'tensor {tensor_list[0].shape}')
            stacked_tensor = torch.stack(tensor_list, dim=0)
            # print(f'stacked {stacked_tensor.shape}')
            return stacked_tensor,idx_list
        else:
            print(f'no imgs key')
            raise KeyError("'imgs' key not found in the HDF5 file.")


def get_srh_base_aug() -> List:
    """Base processing augmentations for all SRH images"""
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    #original srh base aug + Imagenet CLIP aug + hidisc train aug from config file
    preprocess_fn = Compose([
        Resize(size=224, interpolation=Image.BICUBIC), 
        CenterCrop(size=(224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0