
"""HLSS training script.

Copyright (c) 2024 Mohamed Bin Zayed University of Artificial Intelligence. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any
import os
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
import pytorch_lightning as pl
import torchmetrics
from models import CLIPTextClassifier,CLIPVisual,HLSSKL
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders)
from losses.hidisc import HiDiscLoss
from clip.clip import load
import wandb

wandb.init(project="HLSS")

class HiDiscSystem(pl.LightningModule):
    """Lightning system for hlss experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int,freeze_mlp: bool):
        super().__init__()
        self.cf_ = cf
        self.freeze_mlp = freeze_mlp

        if cf["model"]["backbone"] == "RN50":
            bb = partial(CLIPVisual, arch=cf["model"]["backbone"])
        else:
            raise NotImplementedError()

        mlp1 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["patch"],
                      templates=cf["model"]["patch_templates"])
        mlp2 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["slide"],
                      templates=cf["model"]["slide_templates"])
        mlp3 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["patient"],
                      templates=cf["model"]["patient_templates"])

        
        self.model = HLSSKL(bb, mlp1,mlp2,mlp3)
        self.patch_emb = torch.transpose(self.model.proj1.zeroshot_weights,1,0).to("cuda")
        self.slide_emb = torch.transpose(self.model.proj2.zeroshot_weights,1,0).to("cuda")
        self.patient_emb = torch.transpose(self.model.proj3.zeroshot_weights,1,0).to("cuda")

        if self.freeze_mlp:
            for param in self.model.proj1.parameters():
                param.requires_grad = False
            for param in self.model.proj2.parameters():
                param.requires_grad = False
            for param in self.model.proj3.parameters():
                param.requires_grad = False

        if "training" in cf:
            crit_params = cf["training"]["objective"]["params"]

            self.criterion1 = HiDiscLoss(
                lambda_patient=0,
                lambda_slide=0,
                lambda_patch=crit_params["lambda_patch"],
                supcon_loss_params=crit_params["supcon_params"])
            self.criterion2 = HiDiscLoss(
                lambda_patient=0,
                lambda_slide=crit_params["lambda_slide"],
                lambda_patch=0,
                supcon_loss_params=crit_params["supcon_params"])
            self.criterion3 = HiDiscLoss(