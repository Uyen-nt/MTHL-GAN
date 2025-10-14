import os
import random
import torch
import numpy as np
import json

from config import get_generate_args, get_paths
from model import Generator
from datautils.dataloader import load_code_name_map, load_meta_data
from datautils.dataset import DatasetReal
from generation.generate import generate_ehr, get_required_number
from generation.stat_ehr import get_basic_statistics, get_top_k_disease, calc_distance


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path, _, params_path = get_paths(args)
    len_dist, _, _, _, code_map = load_meta_data(dataset_path)
    code_name_map = load_code_name_map(args.data_path)

    # ======================================================
    # 🧩 Kiểm tra hierarchical meta
    # ======================================================
    hier_meta_path = os.path.join(dataset_path, "standard_hier", "hier_meta.json")
    hier_mode = os.path.exists(hier_meta_path)

    if hier_mode:
        print("🔍 Found hierarchical metadata, using dual (diag+proc) generator...")
        with open(hier_meta_path) as f:
            meta = json.load(f)
        code_num = meta["V"]
        Vd, Vp = meta["Vd"], meta["Vp"]
    else:
        code_num = len(code_map)
        Vd, Vp = code_num, 0

    # ======================================================
    # 🧩 Build inverse code map (đầy đủ cho hier mode)
    # ======================================================
    if isinstance(list(code_map.keys())[0], str):
        inv_code_map = {v: k for k, v in code_map.items()}
    else:
        inv_code_map = code_map

    icode_map = {}
    for i in range(Vd):
        icode_map[i] = inv_code_map.get(i, f"DIAG_{i}")
    for i in range(Vd, Vd + Vp):
        icode_map[i] = f"PROC_{i - Vd}"

    # ======================================================
    # 📦 Load real data (đúng folder)
    # ======================================================
    if hier_mode:
        data_dir = os.path.join(dataset_path, "standard_hier", "real_data")
    else:
        data_dir = os.path.join(dataset_path, "standard", "real_data")

    dataset_real = DatasetReal(data_dir)
    len_dist = torch.from_numpy(len_dist).to(device)
    max_len = dataset_real.train_set.data[0].shape[1]

    # ======================================================
    # 🔧 Load generator checkpoint
    # ======================================================
    if args.use_iteration == -1:
        param_file_name = 'generator.pt'
    else:
        param_file_name = f'generator.{args.use_iteration}.pt'

    generator = Generator(code_num=code_num,
                          hidden_dim=args.g_hidden_dim,
                          attention_dim=args.g_attention_dim,
                          max_len=max_len,
                          device=device).to(device)
    generator.load(params_path, param_file_name)

    # ======================================================
    # 🧬 Generate samples
    # ======================================================
    fake_x, fake_lens = generate_ehr(generator, args.number, len_dist, args.batch_size)

    """------------------------get statistics------------------------"""
    real_x, real_lens = dataset_real.train_set.data
    print('real data')
    n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(real_x, real_lens)
    print(f'{args.number} samples -- code types: {n_types} -- code num: {n_codes} '
          f'-- avg code num: {avg_code_num:.4f}, avg visit len: {avg_visit_num:.4f}')
    get_top_k_disease(real_x, real_lens, icode_map, code_name_map, top_k=10)

    print('fake data')
    n_types, n_codes, n_visits, avg_code_num, avg_visit_num = get_basic_statistics(fake_x, fake_lens)
    print(f'{args.number} samples -- code types: {n_types} -- code num: {n_codes} '
          f'-- avg code num: {avg_code_num:.4f}, avg visit len: {avg_visit_num:.4f}')
    get_top_k_disease(fake_x, fake_lens, icode_map, code_name_map, top_k=10)

    jsd_v, jsd_p, nd_v, nd_p = calc_distance(real_x, real_lens, fake_x, fake_lens, code_num)
    print(f'JSD_v: {jsd_v:.4f}, JSD_p: {jsd_p:.4f}, ND_v: {nd_v:.4f}, ND_p: {nd_p:.4f}')
    """------------------------get statistics------------------------"""

    # ======================================================
    # 💾 Save synthetic dataset
    # ======================================================
    synthetic_path = os.path.join(args.result_path, f'synthetic_{args.dataset}.npz')
    np.savez_compressed(synthetic_path, x=fake_x, lens=fake_lens)
    print(f'✅ Saved synthetic data: {synthetic_path}')

    # ======================================================
    # 🩺 Nếu hierarchical: tách diag/proc để tiện xử lý
    # ======================================================
    if hier_mode:
        print("🩺 Splitting hierarchical fake data (diag/proc)...")
        diag = fake_x[:, :, :Vd]
        proc = fake_x[:, :, Vd:]
        hier_path = os.path.join(args.result_path, f"synthetic_{args.dataset}_hier.npz")
        np.savez_compressed(hier_path, diag=diag, proc=proc, lens=fake_lens)
        print(f"✅ Saved hierarchical synthetic data: {hier_path}")
        print(f"   → diag shape: {diag.shape}, proc shape: {proc.shape}")

    # ======================================================
    # Optional: estimate required samples for upper bound
    # ======================================================
    get_required_number(generator, len_dist, args.batch_size, args.upper_bound)


if __name__ == '__main__':
    args = get_generate_args()
    generate(args)
