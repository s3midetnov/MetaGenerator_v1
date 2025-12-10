import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import List, Dict, Any, Optional


class GeneratorModule(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            lr: float,
            warmup_steps: int,
            num_beams: int,
            eval_num_retrieved: int,
            eval_num_workers: int,
            eval_num_gpus: int,
            eval_num_theorems: int,
            max_inp_seq_len: int,
            max_oup_seq_len: int,
            length_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_workers = eval_num_workers
        self.eval_num_gpus = eval_num_gpus
        self.eval_num_theorems = eval_num_theorems
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

    @classmethod
    def load(
        cls, ckpt_path: str, device, freeze: bool
    ):
        model = cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
        return model


    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

  ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    # def configure_optimizers(self) -> Dict[str, Any]:
    #     return get_optimizers(
    #         self.parameters(), self.trainer, self.lr, self.warmup_steps
    #     )
    def validation_step(self, batch, batch_idx):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_val",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        return loss
