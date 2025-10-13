import torch
from torch import nn

from model.utils import MaskedAttention
from model.halo_model import HALOModel


class HALOGeneratorCore(nn.Module):
    """
    Core generator dựa trên HALO thay cho GRU.
    Sinh sequence multi-hot bằng Transformer + autoregressive head.
    """
    def __init__(self, halo_model, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.halo = halo_model                     # HALOModel đã khởi tạo sẵn
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        # ép chiều ẩn HALO (n_embd) về hidden_dim của MTGAN nếu khác
        self.proj = nn.Linear(self.halo.transformer.n_embd, hidden_dim)

    def forward(self, target_codes, lens):
        """
        Trả về:
            probs   : (B, T, V)  xác suất mã
            hiddens : (B, T, H)  ẩn (đã ép chiều)
        """
        B = len(lens)
        V = self.code_num
        T = self.max_len
        device = self.device

        # tạo input trống
        x = torch.zeros(B, T, V, device=device)
        # neo target code đầu tiên vào visit[0]
        x[torch.arange(B), 0, target_codes] = 1.0

        # chạy HALO
        hidden_states = self.halo.transformer(x)                     # (B, T, E)
        code_probs = self.halo.ehr_head(hidden_states, x).sigmoid()  # (B, T-1, V)

        # đệm timestep đầu cho khớp T
        probs = torch.zeros(B, T, V, device=device)
        probs[:, 0, :] = x[:, 0, :]
        probs[:, 1:, :] = code_probs

        # chiếu hidden về hidden_dim cho Critic
        hiddens = self.proj(hidden_states)
        return probs, hiddens


class SmoothCondition(nn.Module):
    """
    Giữ nguyên để tăng xác suất mã mục tiêu.
    Có thể bỏ nếu HALO đã học tốt quan hệ này.
    """
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)

    def forward(self, x, lens, target_codes):
        score = self.attention(x, lens)
        score_tensor = torch.zeros_like(x)
        score_tensor[torch.arange(len(x)), :, target_codes] = score
        x = x + score_tensor
        x = torch.clip(x, max=1)
        return x
