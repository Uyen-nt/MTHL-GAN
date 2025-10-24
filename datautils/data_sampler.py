import torch
import numpy as np


class CodeSampleIter:
    def __init__(self, code, samples, shuffle=True):
        self.code = code
        self.samples = samples
        self.length = len(samples)
        self.current_index = 0
        if shuffle:
            np.random.shuffle(self.samples)

    def __next__(self):
        if self.length == 0:
            raise StopIteration
        sample = self.samples[self.current_index]
        self.current_index = (self.current_index + 1) % self.length
        return sample


class DataSampler:
    def __init__(self, ehr_data, lens, code_num, device=None):
        """
        ehr_data: numpy array [N, T, V]
        lens: numpy array [N]
        code_num: tổng số mã (VD: 2869 hoặc 3686)
        """
        self.ehr_data = ehr_data
        self.lens = lens
        self.code_num = code_num
        self.device = device
        self.size = len(ehr_data)

        self.code_samples = self._get_code_sample_map()

    def _get_code_sample_map(self):
        print(f'🧩 Building EHR data sampler (code_num={self.code_num}) ...')
        code_sample_map = [set() for _ in range(self.code_num)]

        # Duyệt từng bệnh nhân và visit
        for i, (sample, len_i) in enumerate(zip(self.ehr_data, self.lens)):
            for t in range(len_i):
                visit = sample[t]
                codes = np.where(visit > 0)[0]
                for code in codes:
                    if code < self.code_num:
                        code_sample_map[code].add(i)
                    else:
                        print(f"⚠️ Warning: code {code} >= code_num={self.code_num}, skipped")

        # Tạo iterator
        code_samples = [None] * self.code_num
        for code in range(self.code_num):
            samples = list(code_sample_map[code])
            code_samples[code] = CodeSampleIter(code, samples) if samples else CodeSampleIter(code, [])
        print(f"✅ Sampler built for {self.code_num} codes")
        return code_samples

    def sample(self, target_codes):
        valid_lines = []
        valid_codes = []
    
        for code in target_codes:
            sampler = self.code_samples[code]
            try:
                line = next(sampler)
                valid_lines.append(line)
                valid_codes.append(code)
            except StopIteration:
                # Code chưa có sample nào -> skip
                continue
    
        # Nếu số lượng valid < yêu cầu, bổ sung ngẫu nhiên
        if len(valid_lines) < len(target_codes):
            needed = len(target_codes) - len(valid_lines)
            extra_idx = np.random.choice(len(self.ehr_data), size=needed, replace=True)
            valid_lines.extend(extra_idx)
            valid_codes.extend(np.random.choice(target_codes, size=needed, replace=True))
    
        # Cắt về cùng độ dài
        min_len = min(len(valid_lines), len(valid_codes))
        valid_lines = valid_lines[:min_len]
        valid_codes = valid_codes[:min_len]
    
        # --- tạo batch thật ---
        data, lens = self.ehr_data[valid_lines], self.lens[valid_lines]
        data = torch.from_numpy(data).to(self.device, dtype=torch.float)
        lens = torch.from_numpy(lens).to(self.device, torch.long)
    
        # Đảm bảo đồng bộ batch
        target_codes = torch.tensor(valid_codes, dtype=torch.long, device=self.device)
    
        return data, lens, target_codes



def get_train_sampler(train_loader, device, code_num=None):
    """
    Tự động lấy code_num từ shape nếu không truyền vào
    """
    data_tuple = train_loader.dataset.data
    if isinstance(data_tuple, (list, tuple)):
        x = data_tuple[0]
        lens = data_tuple[1]
    else:
        x = data_tuple
        lens = None
    # Tự động xác định code_num nếu chưa truyền
    if code_num is None:
        code_num = x.shape[-1]
        print(f"📊 Auto-detected code_num = {code_num}")
    data_sampler = DataSampler(x, lens, code_num, device)
    return data_sampler
