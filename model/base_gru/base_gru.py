import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class BaseHALO(BaseModel):
    def __init__(self, halo_model, max_len, hidden_dim):
        super().__init__(param_file_name="base_halo.pt")
        self.halo = halo_model
        self.max_len = max_len
        self.proj = nn.Linear(halo_model.transformer.n_embd, hidden_dim)  # ép về cùng kích thước GRU cũ

    def forward(self, x):
        # HALO trả (B, T-1, V) → đệm timestep đầu để khớp PredictNextLoss
        with torch.no_grad():
            code_probs = self.halo(x)
        B, T, V = x.shape
        out = torch.zeros(B, T, V, device=x.device)
        out[:, 0, :] = x[:, 0, :]
        out[:, 1:, :] = code_probs
        return out

    def calculate_hidden(self, x, lens):
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(-1)
            hidden = self.halo.transformer(x)  # (B, T, E)
            hidden = self.proj(hidden) * mask  # ép E→hidden_dim
            return hidden
