
"""Common modules for HiDisc + OpenSRH training and evaluation.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import math
import logging
import argparse
from shutil import copy2
from datetime import datetime
from functools import partial
from typing import Tuple, Dict, Optional, Any

import uuid

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchvision.transforms import Compose

import pytorch_lightning as pl
