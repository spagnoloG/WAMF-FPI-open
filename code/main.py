from pytorch_lightning.cli import LightningCLI
from model import CrossViewLocalizationModel
from data_module import DataModule

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=CrossViewLocalizationModel,
        datamodule_class=DataModule,
    )
