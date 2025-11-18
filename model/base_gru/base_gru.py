import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


# class BaseHALO(BaseModel):
#     def __init__(self, halo_model, max_len, hidden_dim):
#         super().__init__(param_file_name="base_halo.pt")
#         self.halo = halo_model
#         self.max_len = max_len
#         self.proj = nn.Linear(halo_model.transformer.n_embd, hidden_dim)  # ép về cùng kích thước GRU cũ

#     def forward(self, x):
#         # HALO trả (B, T-1, V) → đệm timestep đầu để khớp PredictNextLoss
#         code_probs = self.halo(x)
#         B, T, V = x.shape
#         out = torch.zeros(B, T, V, device=x.device)
#         out[:, 0, :] = x[:, 0, :]
#         out[:, 1:, :] = code_probs
#         return out

#     def calculate_hidden(self, x, lens):
#         with torch.no_grad():
#             mask = sequence_mask(lens, self.max_len).unsqueeze(-1).to(x.device)
#             hidden = self.halo.transformer(x)  # (B, T', E)
#             # 🔧 Pad hidden nếu HALO trả T' < max_len
#             if hidden.size(1) < self.max_len:
#                 pad_len = self.max_len - hidden.size(1)
#                 pad = torch.zeros(hidden.size(0), pad_len, hidden.size(2), device=x.device)
#                 hidden = torch.cat([hidden, pad], dim=1)
                
#             hidden = self.proj(hidden) * mask  # ép E→hidden_dim và áp mask
#             return hidden

class BaseHALO(BaseModel):
    def __init__(self, halo_model, max_len, hidden_dim):
        super().__init__(param_file_name="base_halo.pt")
        self.halo = halo_model
        self.max_len = max_len
        self.hidden_dim = hidden_dim  # Thêm để đảm bảo consistency
        self.proj = nn.Linear(halo_model.transformer.n_embd, hidden_dim)

    def forward(self, x):
        """Dự đoán visit tiếp theo - tương thích với PredictNextLoss"""
        code_probs = self.halo(x)  # [B, T-1, V]
        
        # Đệm để khớp shape [B, T, V]
        B, T, V = x.shape
        output = torch.zeros(B, T, V, device=x.device)
        output[:, 0, :] = x[:, 0, :]  # Visit đầu giữ nguyên
        output[:, 1:, :] = code_probs  # Các visit sau là dự đoán
        
        return output

    def calculate_hidden(self, x, lens):
        """Tính hidden states cho Critic - FIXED"""
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(-1).to(x.device)
            
            # HALO transformer hidden states
            hidden_states = self.halo.transformer(x)  # [B, T, E]
            
            # Đảm bảo shape [B, max_len, hidden_dim]
            if hidden_states.size(1) < self.max_len:
                pad_len = self.max_len - hidden_states.size(1)
                pad = torch.zeros(hidden_states.size(0), pad_len, hidden_states.size(2), 
                                device=x.device)
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            
            # Project về hidden_dim và áp mask
            hidden = self.proj(hidden_states) * mask
            return hidden

    def pretrain_forward(self, x):
        """Forward cho pretraining (không đệm)"""
        return self.halo(x)  # Trả về [B, T-1, V] cho pretraining
