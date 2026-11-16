"""Trainer for protein inverse folding models with PyTorch AMP."""

from typing import Dict, Any, Optional, Tuple, List
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pif.data.constants import PAD_IDX
from pif.evaluation.metrics import compute_perplexity, compute_inverse_folding_metrics


class InverseFoldingTrainer:
    """Trainer managing model optimization, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 15))
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
        self.use_amp = train_cfg.get("amp", True) and torch.cuda.is_available()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-5
            )
        else:
            self.optimizer = None
            self.scheduler = None

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
            "val_recovery": []
        }

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked cross-entropy loss with label smoothing."""
        valid_mask = mask & (targets < 20) & (targets != PAD_IDX)
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        return F.cross_entropy(valid_logits, valid_targets, label_smoothing=self.label_smoothing)

    def train_epoch(self) -> Tuple[float, float]:
        if self.optimizer is None:
            return 0.0, 0.0
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.train_loader:
            coords = batch["coords"].to(self.device)
            edge_attr = batch["edge_attr"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            mask = batch["mask"].to(self.device)
            seq_tokens = batch["seq_tokens"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(coords, edge_attr, edge_index, mask, seq_tokens)
                loss = self._compute_loss(logits, seq_tokens, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            valid_count = int(torch.sum(mask & (seq_tokens < 20)).item())
            total_loss += loss.item() * valid_count
            total_tokens += valid_count

        avg_loss = total_loss / max(total_tokens, 1)
        train_ppl = float(np.exp(min(avg_loss, 20.0)))
        return avg_loss, train_ppl

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        all_pred_tokens = []
        all_targ_tokens = []
        all_ss = []
        all_env = []
        all_cath = []

        for batch in loader:
            coords = batch["coords"].to(self.device)
            edge_attr = batch["edge_attr"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            mask = batch["mask"].to(self.device)
            seq_tokens = batch["seq_tokens"].to(self.device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(coords, edge_attr, edge_index, mask, seq_tokens)
                loss = self._compute_loss(logits, seq_tokens, mask)

            valid_count = int(torch.sum(mask & (seq_tokens < 20)).item())
            total_loss += loss.item() * valid_count
            total_tokens += valid_count

            # Argmax predictions for fast metric tracking
            preds = torch.argmax(logits, dim=-1)  # (B, L)

            mask_np = mask.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targs_np = seq_tokens.cpu().numpy()

            for b in range(len(batch["lengths"])):
                L = batch["lengths"][b].item()
                all_pred_tokens.append(preds_np[b, :L])
                all_targ_tokens.append(targs_np[b, :L])
                all_ss.append(batch["sec_structs"][b][:L])
                all_env.append(batch["environments"][b][:L])
                all_cath.append(int(batch["cath_classes"][b]))

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = float(np.exp(min(avg_loss, 20.0)))

        metrics = compute_inverse_folding_metrics(
            pred_tokens_list=all_pred_tokens,
            target_tokens_list=all_targ_tokens,
            sec_structs_list=all_ss,
            environments_list=all_env,
            cath_classes_list=all_cath,
            ppl_value=ppl
        )

        recovery = metrics["sequence_recovery"]
        return avg_loss, ppl, recovery, metrics

    def fit(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        best_recovery = -1.0
        best_metrics = {}

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_ppl = self.train_epoch()
            val_loss, val_ppl, val_recovery, val_metrics = self.evaluate(self.val_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.time() - t0

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_ppl"].append(train_ppl)
            self.history["val_loss"].append(val_loss)
            self.history["val_ppl"].append(val_ppl)
            self.history["val_recovery"].append(val_recovery)

            if val_recovery > best_recovery:
                best_recovery = val_recovery
                best_metrics = val_metrics
                if save_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_recovery": val_recovery,
                        "val_ppl": val_ppl,
                        "config": self.config
                    }, save_path)

            print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss:.4f} PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} Rec: {val_recovery:.2f}% | {elapsed:.1f}s")

        return best_metrics

