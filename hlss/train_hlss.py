
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
import pytorch_lightning as pl
import torchmetrics
from models import HLSSContrastiveLearningNetwork, CLIPTextClassifier,CLIPVisual
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders)
from losses.hidisc import HiDiscLoss
from clip.clip import load
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
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

        mlp = partial(CLIPTextClassifier,
                      n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["data_classes"],
                      templates=cf["model"]["templates"])
        
        self.model = HLSSContrastiveLearningNetwork(bb, mlp)

        if self.freeze_mlp:
            for param in self.model.proj.parameters():
                param.requires_grad = False

        if "training" in cf:
            crit_params = cf["training"]["objective"]["params"]
            self.criterion = HiDiscLoss(
                lambda_patient=crit_params["lambda_patient"],
                lambda_slide=crit_params["lambda_slide"],
                lambda_patch=crit_params["lambda_patch"],
                supcon_loss_params=crit_params["supcon_params"])
            self.train_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            }) # yapf: disable
            self.val_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            })  #yapf: disable
        else:
            self.criterion = self.train_loss = self.val_loss = None

        self.num_it_per_ep_ = num_it_per_ep

    def forward(self, batch):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        return pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

    def training_step(self, batch, _):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        log_partial = partial(self.log,
                              on_step=True,
                              on_epoch=True,
                              batch_size=bs,
                              sync_dist=True,
                              rank_zero_only=True)
        for k in self.train_loss:
            log_partial(f"train/{k}", losses[k])
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["sum_loss"]

    def validation_step(self, batch, batch_idx):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        for k in self.val_loss:
            self.val_loss[k].update(losses[k], weight=bs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        self.model.eval()
        assert len(batch["image"].shape) == 4
        out = self.model.bb(batch["image"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }

    def on_train_epoch_end(self):
        for k in self.train_loss:
            train_loss_k = self.train_loss[k].compute()
            self.log(f"train/{k}_manualepoch",
                     train_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"train/{k}_manualepoch {train_loss_k}")
            wandb.log({f"train/{k}": train_loss_k})
            self.train_loss[k].reset()

    def on_validation_epoch_end(self):
        for k in self.val_loss:
            val_loss_k = self.val_loss[k].compute()
            self.log(f"val/{k}_manualepoch",
                     val_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"val/{k}_manualepoch {val_loss_k}")
            wandb.log({f"val/{k}": val_loss_k})
            self.val_loss[k].reset()


    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # check if use a learn rate scheduler
        sched_func = get_scheduler_func(self.cf_, self.num_it_per_ep_)
        if not sched_func:
            return opt

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sched_func(opt),
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config


def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    train_loader, valid_loader = get_dataloaders(cf)
    system_func = HiDiscSystem

    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    print(f'noof train minibatches {num_it_per_ep}')
    print(f'noof val minibatches {len(valid_loader)}')
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()
        print(f'num_it_per_ep after distribution {num_it_per_ep}')

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        every_n_epochs=cf["training"]["eval_ckpt_ep_freq"],
        filename="ckpt-epoch{epoch}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        default_root_dir=exp_root,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False,
                                           static_graph=True),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[epoch_ckpt, lr_monitor],
        max_epochs=cf["training"]["num_epochs"],
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        num_nodes=1)
    
    exp = HiDiscSystem(cf, num_it_per_ep,freeze_mlp=False)

    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

    # uncomment if resuming from a given ckpt
    # trainer.fit(exp,
    #             train_dataloaders=train_loader,
    #             val_dataloaders=valid_loader, ckpt_path = "")


if __name__ == '__main__':
    main()