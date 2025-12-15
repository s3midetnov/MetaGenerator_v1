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

# import pytorch_lightning as pl
# import torch
# from torch import nn
# from transformers import AutoModelForSeq2SeqLM
# from transformers import T5ForConditionalGeneration, AutoTokenizer
# from typing import List, Dict, Any, Optional
#
#
# class GeneratorModule(pl.LightningModule):
#     def __init__(
#             self,
#             model_name: str,
#             lr: float,
#             warmup_steps: int,
#             num_beams: int,
#             eval_num_retrieved: int,
#             eval_num_workers: int,
#             eval_num_gpus: int,
#             eval_num_theorems: int,
#             max_inp_seq_len: int,
#             max_oup_seq_len: int,
#             length_penalty: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.warmup_steps = warmup_steps
#         self.num_beams = num_beams
#         self.length_penalty = length_penalty
#         self.eval_num_retrieved = eval_num_retrieved
#         self.eval_num_workers = eval_num_workers
#         self.eval_num_gpus = eval_num_gpus
#         self.eval_num_theorems = eval_num_theorems
#         self.max_inp_seq_len = max_inp_seq_len
#         self.max_oup_seq_len = max_oup_seq_len
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.generator = T5ForConditionalGeneration.from_pretrained(model_name)
#
#     @classmethod
#     def load(
#         cls, ckpt_path: str, device, freeze: bool
#     ):
#         model = cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
#         return model
#
#
#     def forward(
#         self,
#         state_ids: torch.Tensor,
#         state_mask: torch.Tensor,
#         tactic_ids: torch.Tensor,
#     ) -> torch.Tensor:
#         # Keep forward returning loss for Lightning's default flow
#         return self.generator(
#             input_ids=state_ids,
#             attention_mask=state_mask,
#             labels=tactic_ids,
#         ).loss
#
#     ############
#     # Logging #
#     ############
#
#     # def _log_io_texts(
#     #         self,
#     #         split: str,
#     #         state_ids: torch.LongTensor,
#     #         tactic_ids: torch.LongTensor,
#     # ) -> None:
#     #     inp = self.tokenizer.decode(state_ids[0], skip_special_tokens=True)
#     #     oup_ids = torch.where(
#     #         tactic_ids[0] == -100, self.tokenizer.pad_token_id, tactic_ids[0]
#     #     )
#     #     oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)
#     #     self.logger.log_text(
#     #         f"{split}_samples",
#     #         ["state", "tactic"],
#     #         [[inp, oup]],
#     #         step=self.global_step,
#     #     )
#
#     ############
#     # Training #
#     ############
#
#     def training_step(self, batch, batch_idx: int):
#         # Run the model to obtain loss and logits for diagnostics
#         outputs = self.generator(
#             input_ids=batch["state_ids"],
#             attention_mask=batch["state_mask"],
#             labels=batch["tactic_ids"],
#         )
#         loss = outputs.loss
#
#         self.log(
#             "loss_train",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             sync_dist=True,
#             batch_size=batch["state_ids"].size(0),
#         )
#
#         # Diagnostics: entropy of token distribution at supervised positions
#         try:
#             with torch.no_grad():
#                 logits = outputs.logits  # (B, T, V)
#                 labels = batch["tactic_ids"]  # (B, T)
#                 mask = labels != -100
#                 if mask.any():
#                     log_probs = logits.log_softmax(dim=-1)
#                     probs = log_probs.exp()
#                     token_entropy = -(probs * log_probs).sum(-1)  # (B, T)
#                     avg_entropy = (token_entropy[mask]).mean()
#                     self.log(
#                         "logits_entropy_train",
#                         avg_entropy,
#                         on_step=True,
#                         on_epoch=True,
#                         sync_dist=True,
#                         batch_size=batch["state_ids"].size(0),
#                     )
#         except Exception:
#             pass
#
#         # Diagnostics: input/target average lengths
#         try:
#             state_len = batch["state_mask"].sum(dim=1).float().mean()
#             tactic_mask = (batch["tactic_ids"] != -100).sum(dim=1).float().mean()
#             self.log("avg_state_len_train", state_len, on_step=True, on_epoch=True, sync_dist=True)
#             self.log("avg_tactic_len_train", tactic_mask, on_step=True, on_epoch=True, sync_dist=True)
#         except Exception:
#             pass
#
#         # Diagnostics: unique input ratio within batch (helps detect duplicate/noisy batches)
#         try:
#             states = batch.get("state")
#             if states is not None and len(states) > 0:
#                 unique_ratio = float(len(set(states)))/float(len(states))
#                 self.log("unique_states_ratio_train", unique_ratio, on_step=True, on_epoch=False, sync_dist=True)
#         except Exception:
#             pass
#
#         # Log current LR
#         try:
#             if self.trainer is not None and len(self.trainer.optimizers) > 0:
#                 lr = self.trainer.optimizers[0].param_groups[0]["lr"]
#                 self.log("lr", lr, on_step=True, on_epoch=False, sync_dist=True)
#         except Exception:
#             pass
#
#         self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
#         return loss
#
#     def configure_optimizers(self):
#         # Use configured learning rate from hyperparameters instead of hardcoded value
#         return torch.optim.Adam(self.parameters(), lr=self.lr)
#     # def configure_optimizers(self) -> Dict[str, Any]:
#     #     return get_optimizers(
#     #         self.parameters(), self.trainer, self.lr, self.warmup_steps
#     #     )
#     # def validation_step(self, batch, batch_idx):
#     #     loss = self(
#     #         batch["state_ids"],
#     #         batch["state_mask"],
#     #         batch["tactic_ids"],
#     #     )
#     #     self.log(
#     #         "loss_val",
#     #         loss,
#     #         on_step=False,
#     #         on_epoch=True,
#     #         sync_dist=True,
#     #         batch_size=len(batch),
#     #     )
#     #     return loss
#
#     def validation_step(self, batch: Dict[str, Any], _) -> None:
#         state_ids = batch["state_ids"]
#         state_mask = batch["state_mask"]
#         tactic_ids = batch["tactic_ids"]
#
#         outputs = self.generator(
#             input_ids=state_ids,
#             attention_mask=state_mask,
#             labels=tactic_ids,
#         )
#         loss = outputs.loss
#         # Align with ModelCheckpoint monitor key "val_loss"
#         self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=state_ids.size(0))
#         self._log_io_texts("val", state_ids, tactic_ids)
#
#         # Generate topk tactic candidates via Beam Search.
#         output = self.generator.generate(
#             input_ids=state_ids,
#             attention_mask=state_mask,
#             max_length=self.max_oup_seq_len,
#             num_beams=self.num_beams,
#             length_penalty=self.length_penalty,
#             do_sample=False,
#             num_return_sequences=self.num_beams,
#             early_stopping=False,
#         )
#         output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
#         batch_size = state_ids.size(0)
#         assert len(output_text) == batch_size * self.num_beams
#         tactics_pred = [
#             output_text[i * self.num_beams : (i + 1) * self.num_beams]
#             for i in range(batch_size)
#         ]
#
#         msg = "\n".join(tactics_pred[0])
#         self.logger.log_text("preds_val", ["tactics"], [[msg]], step=self.global_step)
#
#         # Log the topk accuracies if available in the module
#         if hasattr(self, "topk_accuracies"):
#             for k in range(1, self.num_beams + 1):
#                 try:
#                     topk_acc = self.topk_accuracies[k]
#                     topk_acc(tactics_pred, batch["tactic"])
#                     self.log(
#                         f"top{k}_acc_val",
#                         topk_acc,
#                         on_step=False,
#                         on_epoch=True,
#                         sync_dist=True,
#                     )
#                 except Exception:
#                     pass
#
#         # Additional validation diagnostics (entropy and lengths)
#         try:
#             with torch.no_grad():
#                 logits = outputs.logits
#                 labels = tactic_ids
#                 mask = labels != -100
#                 if mask.any():
#                     log_probs = logits.log_softmax(dim=-1)
#                     probs = log_probs.exp()
#                     token_entropy = -(probs * log_probs).sum(-1)
#                     avg_entropy = (token_entropy[mask]).mean()
#                     self.log("logits_entropy_val", avg_entropy, on_step=False, on_epoch=True, sync_dist=True)
#             state_len = state_mask.sum(dim=1).float().mean()
#             tactic_len = (tactic_ids != -100).sum(dim=1).float().mean()
#             self.log("avg_state_len_val", state_len, on_step=False, on_epoch=True, sync_dist=True)
#             self.log("avg_tactic_len_val", tactic_len, on_step=False, on_epoch=True, sync_dist=True)
#             states = batch.get("state")
#             if states is not None and len(states) > 0:
#                 unique_ratio = float(len(set(states)))/float(len(states))
#                 self.log("unique_states_ratio_val", unique_ratio, on_step=False, on_epoch=True, sync_dist=True)
#         except Exception:
#             pass