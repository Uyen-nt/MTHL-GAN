import random

import torch
import numpy as np

from config import get_training_args, get_paths
from model import Generator, Critic, BaseHALO
from model.halo_model import HALOModel
from model.generator.generator import Generator
from trainer import GANTrainer, BaseGRUTrainer
from datautils.dataloader import load_code_name_map, load_meta_data, get_train_test_loader, get_base_gru_train_loader

from types import SimpleNamespace
import json
import os


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, records_path, params_path = get_paths(args)
    
    # ======================================================
    # 🧩 Hierarchical mode: đọc thông tin vocab từ hier_meta.json
    # ======================================================
  
    hier_meta_path = os.path.join(dataset_path, "standard_hier", "hier_meta.json")
    if os.path.exists(hier_meta_path):
        print("🔍 Found hierarchical metadata, loading hier_meta.json ...")
        with open(hier_meta_path) as f:
            meta = json.load(f)
        code_num = meta["V"]
        Vd = meta["Vd"]
        Vp = meta["Vp"]
        print(f"    → Using hierarchical vocab: total={code_num}, diag={Vd}, proc={Vp}")
        hier_mode = True
    else:
        hier_mode = False

    len_dist, code_visit_dist, code_patient_dist, code_adj, code_map = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)
    train_loader, test_loader, max_len = get_train_test_loader(dataset_path, args.batch_size, device)

    # ======================================================
    # 📦 Load đúng dataset: hier hoặc single
    # ======================================================
    if hier_mode:
        hier_data_path = os.path.join(dataset_path, "standard_hier", "real_data")
        print(f"📂 Loading hierarchical dual dataset from: {hier_data_path}")
        train_loader, test_loader, max_len = get_train_test_loader(hier_data_path, args.batch_size, device)
    else:
        print("📂 Loading standard single-diagnosis dataset ...")
        train_loader, test_loader, max_len = get_train_test_loader(
            os.path.join(dataset_path, "standard", "real_data"),
            args.batch_size, device
        )

        
    len_dist = torch.from_numpy(len_dist).to(device)

    config = SimpleNamespace(
        n_layer=args.halo_n_layer,
        n_embd=args.halo_n_embd,               # đồng bộ hidden_dim
        n_head=args.halo_n_head,
        n_ctx=args.halo_n_ctx,
        n_positions=args.halo_n_positions,
        layer_norm_epsilon=args.halo_layer_norm_epsilon,
        total_vocab_size=code_num              # số mã ICD/procedure tổng cộng
    )
    halo_model = HALOModel(config).to(device)
    base_gru = BaseHALO(halo_model, max_len=max_len, hidden_dim=args.g_hidden_dim).to(device)
    try:
        base_gru.load(params_path)
    except FileNotFoundError:
        base_gru_trainloader = get_base_gru_train_loader(dataset_path, args.batch_size, device)
        base_gru_trainer = BaseGRUTrainer(args, base_gru, max_len, base_gru_trainloader, params_path)
        base_gru_trainer.train()
    base_gru.eval()

    generator = Generator(halo_model, code_num=code_num,
                          hidden_dim=args.g_hidden_dim,
                          attention_dim=args.g_attention_dim,
                          max_len=max_len,
                          device=device).to(device)
    critic = Critic(code_num=code_num,
                    hidden_dim=args.c_hidden_dim,
                    generator_hidden_dim=args.g_hidden_dim,
                    max_len=max_len).to(device)

    print('Param number:', count_model_params(generator) + count_model_params(critic))

    trainer = GANTrainer(args,
                         generator=generator, critic=critic, base_gru=base_gru,
                         train_loader=train_loader, test_loader=test_loader,
                         len_dist=len_dist, code_map=code_map, code_name_map=code_name_map,
                         records_path=records_path, params_path=params_path)
    trainer.train()


if __name__ == '__main__':
    args = get_training_args()
    train(args)
