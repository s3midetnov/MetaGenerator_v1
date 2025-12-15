import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer
import scripts.datasetscripts as dp

Example = Dict[str, Any]
Batch = Dict[str, Any]

# Custom dataset module
class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        for thm in tqdm(dp.parse_concatenated_json(data_path)):
            data.append(
                {
                    "exp_type": thm["Expected type"], # = state
                    "expression": thm["Expression"], # = tactic
                    "premises": thm["Premises"], # = premises
                    "context": thm["Context"], # = ?
                }
            )
        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        ex["state"] = dp.format_augmented_goal(
            ex["exp_type"],
            ex["premises"],
            self.max_inp_seq_len,
            self.p_drop if self.is_train else 0.0,
        )

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        state = [ex["state"] for ex in examples]
        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic = [ex["expression"] for ex in examples]
        tokenized_tactic = self.tokenizer(
            tactic,
            padding="longest",
            max_length=self.max_oup_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic_ids = tokenized_tactic.input_ids
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tactic_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self) -> None:
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                os.path.join(self.data_path, "train.json"),
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                os.path.join(self.data_path, "val.json"),
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
