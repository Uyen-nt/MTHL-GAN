import torch
from torch import nn

from model.utils import MaskedAttention
from model.halo_model import HALOModel


class HALOGeneratorCore(nn.Module):
    """
    Core generator dựa trên HALO thay cho GRU.
    Sinh sequence multi-hot bằng Transformer + autoregressive head.
    """
    def __init__(self, halo_model, code_num, hidden_dim, max_len, device=None, sparsity_threshold=0.1):
        super().__init__()
        self.halo = halo_model                     # HALOModel đã khởi tạo sẵn
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device
        self.sparsity_threshold = sparsity_threshold

        # ép chiều ẩn HALO (n_embd) về hidden_dim của MTGAN nếu khác
        self.proj = nn.Linear(self.halo.transformer.n_embd, hidden_dim)

    def apply_sparsity(self, probs, threshold=None):
        """Áp dụng sparsity để giảm số lượng codes"""
        if threshold is None:
            threshold = self.sparsity_threshold
        
        # Cách 1: Top-k sampling
        k = int(probs.shape[-1] * 0.05)  # Giữ lại 5% codes có xác suất cao nhất
        topk_vals, topk_indices = torch.topk(probs, k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, topk_indices, 1.0)
        return probs * mask

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

        # tạo input
        x = torch.zeros(B, T, V, device=device)
        for i in range(B):
            num_codes = torch.randint(1, 8, (1,))  # 1-7 codes mỗi visit
            codes = torch.randperm(V)[:num_codes]
            x[i, 0, codes] = 1.0
        

        # chạy HALO
        hidden_states = self.halo.transformer(x)                     # (B, T, E)
        code_probs = self.halo.ehr_head(hidden_states, x).sigmoid()  # (B, T-1, V)
        code_probs = self.apply_sparsity(code_probs)

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
