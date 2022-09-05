import time
import pytorch_lightning as pl
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
import torch
class TimerCallback(pl.Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_fit_start at {tic}")

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_fit_end at {tic}")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_train_start at {tic}")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_train_epoch_start at {tic}")

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        tic = time.perf_counter()
        print(f"on_train_batch_start at {tic}")

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        tic = time.perf_counter()
        print(f"backward at {tic}")

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_after_backward at {tic}")

    def on_before_optimizer_step(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer, opt_idx: int
    ) -> None:
        tic = time.perf_counter()
        print(f"optimizer_step at {tic}")

    def on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer) -> None:
        tic = time.perf_counter()
        print(f"on_before_zero_grad at {tic}")

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        tic = time.perf_counter()
        print(f"optimizer_zero_grad at {tic}")

    def training_step_end(self, batch_parts):
        tic = time.perf_counter()
        print(f"training_step_end at {tic}")

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            # outputs: STEP_OUTPUT,
            outputs,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        tic = time.perf_counter()
        print(f"on_train_batch_end at {tic}")

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_validation_epoch_start at {tic}")

    def on_validation_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        tic = time.perf_counter()
        print(f"on_validation_batch_start at {tic}")

    def validation_step_end(self, batch_parts):
        tic = time.perf_counter()
        print(f"validation_step_end at {tic}")

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            # outputs: Optional[STEP_OUTPUT],
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        tic = time.perf_counter()
        print(f"on_validation_batch_end at {tic}")

    def validation_epoch_end(self, validation_step_outputs):
        tic = time.perf_counter()
        print(f"validation_epoch_end at {tic}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        tic = time.perf_counter()
        print(f"on_validation_epoch_end at {tic}")

    def training_epoch_end(self, training_step_outputs):
        # all_preds = torch.stack(training_step_outputs)
        tic = time.perf_counter()
        print(f"training_epoch_end at {tic}")

    def on_train_end(self, trainer, pl_module):
        toc = time.perf_counter()
        print(f"on_train_end at {toc}")