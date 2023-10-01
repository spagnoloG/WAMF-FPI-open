#!/usr/bin/env python3

from pytorch_lightning.cli import LightningCLI
from model import CrossViewLocalizationModel
from data_module import DataModule
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl


class MyCLI(LightningCLI):
    def before_fit(self):
        self.trainer.logger = WandbLogger(project="cross-view-localization-model")
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="hann_loss",
        )
        self.log_predictions_callback = LogPredictionsCallback()

        self.trainer.callbacks.append(self.checkpoint_callback)
        self.trainer.callbacks.append(self.log_predictions_callback)


class LogPredictionsCallback(pl.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            pass

    # def on_train_epoch_end(self, trainer, pl_module):
    #    pl_module.on_end(mode="train")

    # def on_validation_epoch_end(self, trainer, pl_module):
    #    pl_module.on_end(mode="val")


if __name__ == "__main__":
    cli = MyCLI(
        model_class=CrossViewLocalizationModel,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True},
    )
