import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os


class SelfSupervisedTrainer:
    """
    Simple masked-code modeling trainer for BaseHALO.
    """
    def __init__(self, base_halo, dataloader, device, params_path,
                 mask_ratio=0.15, lr=1e-4, epochs=10):
        self.base_halo = base_halo
        self.dataloader = dataloader
        self.device = device
        self.params_path = params_path
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.epochs = epochs
        self.opt = torch.optim.Adam(self.base_halo.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def mask_inputs(self, x):
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0
        return x_masked, mask

    def train(self):
        print(f"\n🎯 Self-supervised warm-up: epochs={self.epochs}, mask_ratio={self.mask_ratio}")
        self.base_halo.train()
        for epoch in range(1, self.epochs + 1):
            losses = []
            for x, _ in tqdm(self.dataloader, desc=f"Epoch {epoch}"):
                x = x.to(self.device, dtype=torch.float)
                x_masked, mask = self.mask_inputs(x)

                logits = self.base_halo(x_masked)  # [B, T, V]
                loss = self.criterion(logits[mask], x[mask])

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        torch.save(self.base_halo.state_dict(), os.path.join(self.params_path, "base_halo_warmup.pt"))
        print("✅ Warm-up done & saved to base_halo_warmup.pt")

