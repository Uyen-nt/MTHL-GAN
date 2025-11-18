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
    Load train/test dataloader cho cả 2 chế độ
    """

    # --- Xác định đường dẫn thật ---
    if "standard_hier" in dataset_path:
        data_dir = dataset_path
        print(f"📦 [Dual Hierarchical] Using dataset at: {data_dir}")
    else:
        data_dir = os.path.join(dataset_path, "standard", "real_data")
        print(f"📦 [Single Diagnosis] Using dataset at: {data_dir}")

    # 🎯 KIỂM TRA VÀ ƯU TIÊN BALANCED DATA
    balanced_train_path = os.path.join(data_dir, "train_balanced.npz")
    standard_train_path = os.path.join(data_dir, "train.npz")
    
    if os.path.exists(balanced_train_path):
        print(f"   🎯 Using BALANCED training data!")
        # Tạo dataset từ balanced data
        dataset = _create_balanced_dataset(data_dir, device)
    elif os.path.exists(standard_train_path):
        print(f"   📂 Using STANDARD training data")
        dataset = DatasetReal(data_dir, device=device)
    else:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    # --- Tạo DataLoader ---
    train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)

    # --- Kiểm tra và in kích thước ---
    sample_x, _ = next(iter(train_loader))
    print(f"✅ Sample batch shape: {tuple(sample_x.shape)}")

    max_len = sample_x.shape[1]
    return train_loader, test_loader, max_len


def _create_balanced_dataset(data_dir, device):
    """Tạo dataset từ balanced data (nếu có) hoặc standard data"""
    class FlexibleDatasetReal:
        def __init__(self, data_dir, device):
            self.device = device
            
            # 🎯 ƯU TIÊN: balanced data trước
            balanced_train_path = os.path.join(data_dir, "train_balanced.npz")
            standard_train_path = os.path.join(data_dir, "train.npz")
            test_path = os.path.join(data_dir, "test.npz")
            
            # Load train data (ưu tiên balanced)
            if os.path.exists(balanced_train_path):
                print("   🎯 Loading BALANCED train data")
                train_data = np.load(balanced_train_path)
            elif os.path.exists(standard_train_path):
                print("   📂 Loading STANDARD train data")  
                train_data = np.load(standard_train_path)
            else:
                raise FileNotFoundError(f"No train data found in {data_dir}")
                
            self.train_set = self._create_data_tuple(train_data)
            
            # Load test data (luôn dùng standard)
            if os.path.exists(test_path):
                test_data = np.load(test_path)
                self.test_set = self._create_data_tuple(test_data)
                test_data.close()
            else:
                raise FileNotFoundError(f"No test data found in {data_dir}")
            
            train_data.close()
        
        def _create_data_tuple(self, data):
            return (torch.from_numpy(data['x']).to(self.device), 
                    torch.from_numpy(data['lens']).to(self.device))
    
    return FlexibleDatasetReal(data_dir, device)


def get_base_gru_train_loader(dataset_path, batch_size, device):
    data_dir = os.path.join(dataset_path, 'standard', 'real_next')
    
    # 🎯 KIỂM TRA BALANCED REAL_NEXT
    balanced_path = os.path.join(data_dir, "train_balanced.npz")
    if os.path.exists(balanced_path):
        print(f"📦 [BaseHALO - BALANCED] Using balanced real_next data")
        # Tạo dataset custom từ balanced
        dataset = _create_balanced_realnext_dataset(data_dir, device)
    else:
        dataset = DatasetRealNext(data_dir, device=device)
    
    train_loader = DataLoader(dataset.train_set, shuffle=True, batch_size=batch_size)
    return train_loader


def _create_balanced_realnext_dataset(data_dir, device):
    """Tạo real_next dataset từ balanced data"""
    class BalancedDatasetRealNext:
        def __init__(self, data_dir, device):
            self.device = device
            
            # Load balanced train data
            balanced_data = np.load(os.path.join(data_dir, "train_balanced.npz"))
            self.train_set = self._create_data_tuple(balanced_data)
            
            balanced_data.close()
        
        def _create_data_tuple(self, data):
            return (torch.from_numpy(data['x']).to(self.device), 
                    torch.from_numpy(data['lens']).to(self.device),
                    torch.from_numpy(data['y']).to(self.device))
    
    return BalancedDatasetRealNext(data_dir, device)


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
