
"""Evaluation modules and script.

Copyright (c) 2024 Mohamed Bin Zayed University of Artificial Intelligence. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any
import tifffile
import re
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torchvision.transforms import Compose
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy
from torchmetrics.classification import BinaryAccuracy
from datasets.srh_dataset import OpenSRHDataset, TCGADataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug,get_tcga_base_aug
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)
from train_hidisc_tcga import HiDiscSystem

# code for kNN prediction is from the github repo IgorSusmelj/barlowtwins
# https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py
def knn_predict(feature, feature_bank, feature_labels, classes: int,
                knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features from a feature bank.

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature
    """
    # cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


def get_embeddings(cf: Dict[str, Any],
                   exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    if cf["model"]["backbone"] == "resnet50":
        aug_func = get_tcga_base_aug
    elif cf["model"]["backbone"] == "vit":
        aug_func = get_srh_vit_base_aug
    else:
        raise NotImplementedError()

    # get dataset / loader
    train_dset = TCGADataset(data_root=cf["data"]["db_root"],
                                studies="train",
                                transform=Compose(aug_func()),
                                balance_slide_per_class=False)
    train_dset.reset_index()

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    val_dset = TCGADataset(data_root=cf["data"]["db_root"],
                              studies="val",
                              transform=Compose(aug_func()),
                              balance_slide_per_class=False)
    val_dset.reset_index()


    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    # load lightning checkpoint
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_path"])

    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,