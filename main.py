"""Script for training the tactic generator."""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from datamodule import GeneratorDataModule
from model import GeneratorModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
        parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(GeneratorModule, GeneratorDataModule, save_config_kwargs={"overwrite": True})
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
