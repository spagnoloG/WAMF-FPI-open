from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.callbacks import Callback
from model import MyModel
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import shutil

class DataModule(pl.LightningDataModule):
    def prepare_data(self):
        MNIST(root="./data", train=True, download=True)
        MNIST(root="./data", train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        if stage == "fit" or stage is None:
            self.mnist_train = MNIST(root="./data", train=True, transform=transform)
            self.mnist_val = MNIST(root="./data", train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)


cli = LightningCLI(MyModel, DataModule, callback=MoveConfigCallback())
