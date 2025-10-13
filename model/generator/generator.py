import torch

from model.base_model import BaseModel
from model.utils import sequence_mask
from .generator_layers import HALOGeneratorCore, SmoothCondition


class Generator(BaseModel):
    """
    Generator mới dùng HALO thay vì GRU.
    - forward() và sample() giữ nguyên API để tương thích với trainer hiện tại.
    """
    def __init__(self, halo_model, code_num, hidden_dim, attention_dim, max_len, device=None):
        super().__init__(param_file_name='generator.pt')
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        self.noise_dim = hidden_dim  # vẫn giữ để trainer không lỗi
        self.halo_core = HALOGeneratorCore(halo_model, code_num, hidden_dim, max_len, device)
        self.smooth_condition = SmoothCondition(code_num, attention_dim)

    def forward(self, target_codes, lens, noise=None):
        """
        - noise để giữ tương thích với trainer (không dùng trong HALO)
        - trả probs, hiddens
        """
        probs, hiddens = self.halo_core(target_codes, lens)
        probs = self.smooth_condition(probs, lens, target_codes)
        return probs, hiddens

    def sample(self, target_codes, lens, noise=None, return_hiddens=False):
        """
        Sinh sample binary (multi-hot 0/1) giống GRU cũ.
        """
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(-1)
            prob, hiddens = self.forward(target_codes, lens)
            samples = torch.bernoulli(prob).float() * mask
            if return_hiddens:
                hiddens = hiddens * mask
                return samples, hiddens
            return samples

    def get_noise(self, batch_size):
        # để trainer không lỗi, trả random noise (không dùng)
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        return noise

    def get_target_codes(self, batch_size):
        codes = torch.randint(low=0, high=self.code_num, size=(batch_size,))
        return codes
