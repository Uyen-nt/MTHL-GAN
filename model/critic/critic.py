import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask


class Critic(BaseModel):
    def __init__(self, code_num, hidden_dim, generator_hidden_dim, max_len):
        super().__init__(param_file_name='critic.pt')

        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.max_len = max_len

        self.linear = nn.Sequential(
            nn.Linear(code_num + generator_hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hiddens, lens):
        # Giả định: x shape = (B, T, V), hiddens shape = (B, T', H)
        T, T_h = x.size(1), hiddens.size(1)
        
        # Pad hoặc cắt hiddens cho khớp với x
        if T_h < T:
            pad_len = T - T_h
            pad = torch.zeros(hiddens.size(0), pad_len, hiddens.size(2), device=hiddens.device)
            hiddens = torch.cat([hiddens, pad], dim=1)
        elif T_h > T:
            hiddens = hiddens[:, :T, :]
        
        output = torch.cat([x, hiddens], dim=-1)
        output = self.linear(output).squeeze(dim=-1)

        mask = sequence_mask(lens, self.max_len)

        if mask.size(1) != output.size(1):
            # Pad hoặc cắt mask cho khớp với output
            if mask.size(1) < output.size(1):
                pad_len = output.size(1) - mask.size(1)
                pad = torch.ones(mask.size(0), pad_len, device=mask.device)
                mask = torch.cat([mask, pad], dim=1)
            else:
                mask = mask[:, :output.size(1)]
        
        output = output * mask
        output = output.sum(dim=-1)
        # Dùng độ dài thực tế để tránh chia sai nếu cắt
        valid_lens = mask.sum(dim=-1)
        output = output / valid_lens.clamp(min=1)
        return output
