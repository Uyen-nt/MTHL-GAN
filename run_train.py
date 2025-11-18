import random
import torch
import numpy as np
import json
import os
from types import SimpleNamespace

from config import get_training_args, get_paths
from model import Generator, Critic, BaseHALO
from model.halo_model import HALOModel
from model.generator.generator import Generator
from trainer import GANTrainer, BaseGRUTrainer
from trainer.self_supervised_trainer import SelfSupervisedTrainer
from datautils.dataloader import (
    load_code_name_map,
    load_meta_data,
    get_train_test_loader,
    get_base_gru_train_loader,
)


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, records_path, params_path = get_paths(args)

    # ======================================================
    # 🧩 1. Kiểm tra hierarchical mode
    # ======================================================
    hier_meta_path = os.path.join(dataset_path, "standard_hier", "hier_meta.json")
    hier_mode = os.path.exists(hier_meta_path)
    if hier_mode:
        print("🔍 Found hierarchical metadata, loading hier_meta.json ...")
        with open(hier_meta_path) as f:
            meta = json.load(f)
        code_num = meta["V"]
        Vd = meta["Vd"]
        Vp = meta["Vp"]
        print(f"    → Using hierarchical vocab: total={code_num}, diag={Vd}, proc={Vp}")
    else:
        Vd, Vp, code_num = 0, 0, None

    # ======================================================
    # 📦 2. Load metadata & code map
    # ======================================================
    len_dist, code_visit_dist, code_patient_dist, code_adj, code_map = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)

    # Đảo key/value nếu code_map là {str: int}
    if isinstance(list(code_map.keys())[0], str):
        inv_code_map = {v: k for k, v in code_map.items()}
    else:
        inv_code_map = code_map

    # ======================================================
    # 🧩 3. Mở rộng map trong hierarchical mode
    # ======================================================
    if hier_mode:
        print("🧩 Extending code_map for hierarchical (diag + proc)...")
        icode_map = {}
        # Phần bệnh
        for i in range(Vd):
            icode_map[i] = inv_code_map.get(i, f"DIAG_{i}")
        # Phần thủ thuật
        for i in range(Vd, Vd + Vp):
            icode_map[i] = f"PROC_{i - Vd}"
    else:
        icode_map = inv_code_map
        code_num = len(code_adj)

    # ======================================================
    # 📂 4. Load dataset đúng mode
    # ======================================================
    if hier_mode:
        hier_data_path = os.path.join(dataset_path, "standard_hier", "real_data")
        print(f"📂 Loading hierarchical dual dataset from: {hier_data_path}")
        train_loader, test_loader, max_len = get_train_test_loader(hier_data_path, args.batch_size, device)
    else:
        print("📂 Loading standard single-diagnosis dataset ...")
        data_path_std = os.path.join(dataset_path, "standard", "real_data")
        train_loader, test_loader, max_len = get_train_test_loader(data_path_std, args.batch_size, device)

    len_dist = torch.from_numpy(len_dist).to(device)

    # ======================================================
    # 🧠 5. Tạo và Warm-up BaseHALO
    # ======================================================
    config = SimpleNamespace(
        n_layer=args.halo_n_layer,
        n_embd=args.halo_n_embd,
        n_head=args.halo_n_head,
        n_ctx=args.halo_n_ctx,
        n_positions=args.halo_n_positions,
        layer_norm_epsilon=args.halo_layer_norm_epsilon,
        total_vocab_size=code_num,
    )
    
    halo_model = HALOModel(config).to(device)
    base_halo = BaseHALO(halo_model, max_len=max_len, hidden_dim=args.g_hidden_dim).to(device)
    
    # 🧩 Dữ liệu pretrain (real_next từ dual dataset)
    hier_realnext_path = os.path.join(dataset_path, "standard_hier", "real_next", "train.npz")
    if not os.path.exists(hier_realnext_path):
        raise FileNotFoundError(f"❌ Missing hierarchical real_next: {hier_realnext_path}")
    
    from datautils.dataset import DatasetRealNext
    from datautils.dataloader import DataLoader
  
    print(f"📚 Loading hierarchical real_next data from {hier_realnext_path}")
    real_next_dataset = DatasetRealNext(os.path.dirname(hier_realnext_path), device=device)
    train_loader = DataLoader(real_next_dataset.train_set, shuffle=True, batch_size=args.batch_size)
    
    # 🧠 Warm-up HALO trước khi đưa vào Critic
    halo_trainer = SelfSupervisedTrainer(
        base_halo,
        train_loader,
        device,
        params_path,
        mask_ratio=0.15,
        lr=1e-4,
        epochs=args.halo_warmup_epochs if hasattr(args, "halo_warmup_epochs") else 10,
    )
    halo_trainer.train()
    base_halo.eval()


    # ======================================================
    # ⚙️ 6. Generator & Critic
    # ======================================================
    generator = Generator(
        halo_model,
        code_num=code_num,
        hidden_dim=args.g_hidden_dim,
        attention_dim=args.g_attention_dim,
        max_len=max_len,
        device=device,
    ).to(device)

    critic = Critic(
        code_num=code_num,
        hidden_dim=args.c_hidden_dim,
        generator_hidden_dim=args.g_hidden_dim,
        max_len=max_len,
    ).to(device)

    print('Param number:', count_model_params(generator) + count_model_params(critic))

    # ======================================================
    # 🚀 7. Train GAN
    # ======================================================
    trainer = GANTrainer(
        args,
        generator=generator,
        critic=critic,
        base_gru=base_halo,
        train_loader=train_loader,
        test_loader=test_loader,
        len_dist=len_dist,
        code_map=icode_map,
        code_name_map=code_name_map,
        records_path=records_path,
        params_path=params_path,
    )
    trainer.train()


if __name__ == '__main__':
    args = get_training_args()
    train(args)

