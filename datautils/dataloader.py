import os
import pickle

import numpy as np
import pandas as pd

from .dataset import DatasetReal, DatasetRealNext


def infinite_dataloader(dataloader):
    while True:
        for x in dataloader:
            yield x


class DataLoader:
    def __init__(self, dataset, shuffle=True, batch_size=32):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.size = len(dataset)
        self.idx = np.arange(self.size)
        self.n_batches = np.ceil(self.size / batch_size).astype(int)

        self.counter = 0
        if shuffle:
            np.random.shuffle(self.idx)

    def _get_item(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = self.dataset[index]
        return data

    def __next__(self):
        if self.counter >= self.n_batches:
            self.counter = 0
            raise StopIteration
        data = self._get_item(self.counter)
        self.counter += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


def get_train_test_loader(dataset_path, batch_size, device):
    """
    Load train/test dataloader cho cả 2 chế độ:
      - Normal (standard/real_data)
      - Hierarchical dual (standard_hier/real_data)
    """

    # --- Xác định đường dẫn thật ---
    if "standard_hier" in dataset_path:
        data_dir = dataset_path  # đã là folder chứa train/test.npz
        print(f"📦 [Dual Hierarchical] Using dataset at: {data_dir}")
    else:
        data_dir = os.path.join(dataset_path, "standard", "real_data")
        print(f"📦 [Single Diagnosis] Using dataset at: {data_dir}")

    # --- Load dataset ---
    dataset = DatasetReal(data_dir, device=device)

    # --- Tạo DataLoader ---
    train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)

    # --- Kiểm tra và in kích thước ---
    sample_x, _ = next(iter(train_loader))
    print(f"✅ Sample batch shape: {tuple(sample_x.shape)}")

    max_len = sample_x.shape[1]
    return train_loader, test_loader, max_len


def get_base_gru_train_loader(dataset_path, batch_size, device):
    dataset = DatasetRealNext(os.path.join(dataset_path, 'standard', 'real_next'), device=device)
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    return train_loader


def load_meta_data(dataset_path):
    standard_path = os.path.join(dataset_path, 'standard')
    real_data_stat = np.load(os.path.join(standard_path, 'real_data_stat.npz'))
    len_dist, code_visit_dist, code_patient_dist = (real_data_stat['admission_dist'], real_data_stat['code_visit_dist'],
                                                    real_data_stat['code_patient_dist'])
    code_adj = np.load(os.path.join(standard_path, 'code_adj.npz'))['code_adj']
    code_map = pickle.load(open(os.path.join(dataset_path, 'encoded', 'code_map.pkl'), 'rb'))
    return len_dist, code_visit_dist, code_patient_dist, code_adj, code_map


def load_code_name_map(data_path):
    names = pd.read_excel(os.path.join(data_path, 'map.xlsx'), engine='openpyxl')
    code_keys = names['DIAGNOSIS CODE'].tolist()
    name_vals = names['LONG DESCRIPTION'].tolist()
    code_name_map = {k: v for k, v in zip(code_keys, name_vals)}
    return code_name_map
