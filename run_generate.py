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

from model.halo_model import HALOModel
from types import SimpleNamespace

from evaluation_metrics import evaluate_dual_outputs, analyze_visit_distribution, calculate_co_occurrence_metrics, calculate_pairwise_cooccurrence_precision_recall, calculate_code_coverage_metrics


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

    config = SimpleNamespace(
        n_layer=args.halo_n_layer,
        n_embd=args.halo_n_embd,
        n_head=args.halo_n_head,
        n_ctx=args.halo_n_ctx,
        n_positions=args.halo_n_positions,
        layer_norm_epsilon=args.halo_layer_norm_epsilon,
        total_vocab_size=code_num
    )
    
    halo_model = HALOModel(config).to(device)
    
    generator = Generator(halo_model,
                      code_num=code_num,
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
    # 🩺 Nếu hierarchical
    # ======================================================
    if hier_mode:
        print("🧬 HALO hierarchical mode detected — keeping unified (diag+proc) format.")
        hier_path = os.path.join(args.result_path, f"synthetic_{args.dataset}_hier.npz")
        np.savez_compressed(hier_path, x=fake_x, lens=fake_lens)
        print(f"✅ Saved unified hierarchical synthetic data: {hier_path}")
        print(f"   → shape: {fake_x.shape}")

    # ======================================================
    # Optional: estimate required samples for upper bound
    # ======================================================
    get_required_number(generator, len_dist, args.batch_size, args.upper_bound)

    # ======================================================
    # 🧪 EVALUATION MỚI - ĐÁNH GIÁ DUAL OUTPUT
    # ======================================================
    
    print("\n" + "="*60)
    print("🧪 EVALUATION DUAL OUTPUTS (DIAGNOSES + PROCEDURES)")
    print("="*60)
    
    # Load real data để so sánh
    if hier_mode:
        data_dir = os.path.join(dataset_path, "standard_hier", "real_data")
    else:
        data_dir = os.path.join(dataset_path, "standard", "real_data")
    
    dataset_real = DatasetReal(data_dir)
    real_x, real_lens = dataset_real.train_set.data
    
    # 1. Basic evaluation
    dual_results = evaluate_dual_outputs(real_x, real_lens, fake_x, fake_lens, Vd, Vp)
    print("📈 BASIC DUAL METRICS:")
    print(f"  Diagnoses - Real: {dual_results['diag_stats']['real_avg_codes_per_visit']:.2f}, Fake: {dual_results['diag_stats']['fake_avg_codes_per_visit']:.2f}")
    print(f"  Procedures - Real: {dual_results['proc_stats']['real_avg_codes_per_visit']:.2f}, Fake: {dual_results['proc_stats']['fake_avg_codes_per_visit']:.2f}")
    print(f"  Visits with both diag+proc - Real: {dual_results['joint_visits']['real_ratio']:.3f}, Fake: {dual_results['joint_visits']['fake_ratio']:.3f}")
    
    # 2. Detailed analysis
    print("\n🔍 DETAILED VISIT ANALYSIS (10 first patients):")
    analyze_visit_distribution(fake_x, fake_lens, Vd, Vp, sample_size=10)
    
    # 3. Co-occurrence analysis
    # print("\n🔗 CO-OCCURRENCE ANALYSIS:")
    # cooccur_metrics = calculate_co_occurrence_metrics(real_x, fake_x, Vd, Vp)
    # print(f"  Real co-occurring visits: {cooccur_metrics['real_cooccurring_visits']}")
    # print(f"  Fake co-occurring visits: {cooccur_metrics['fake_cooccurring_visits']}")
    # print(f"  Co-occurrence JS Distance: {cooccur_metrics['cooccurrence_js_distance']:.4f}")

    print("\n" + "="*60)
    print("METRICS NÂNG CAO: CODE COVERAGE & RARE CODE RECOVERY")
    print("="*60)
    
    coverage_metrics = calculate_code_coverage_metrics(real_x, real_lens, fake_x, fake_lens, Vd, Vp, rare_threshold=10)
    print(f" Diagnoses  - Unique: {coverage_metrics['diagnosis']['real_unique']} → {coverage_metrics['diagnosis']['fake_unique']} "
          f"({coverage_metrics['diagnosis']['coverage_ratio']:.3f}), Rare recall: {coverage_metrics['diagnosis']['rare_recall']:.3f}")
    print(f" Procedures - Unique: {coverage_metrics['procedure']['real_unique']} → {coverage_metrics['procedure']['fake_unique']} "
          f"({coverage_metrics['procedure']['coverage_ratio']:.3f}), Rare recall: {coverage_metrics['procedure']['rare_recall']:.3f}")

    print("\n" + "="*60)
    print("METRICS NÂNG CAO: CO-OCCURRENCE PRECISION (Top 1000 cặp phổ biến)")
    print("="*60)
    
    pair_metrics = calculate_pairwise_cooccurrence_precision_recall(real_x, fake_x, Vd, Vp, top_k_pairs=1000)
    print(f" Top-1000 real pairs được sinh lại: {pair_metrics['generated_matching_pairs']}/{pair_metrics['top_k']} "
          f"→ Precision = {pair_metrics['cooccurrence_precision']:.3f}")
    print(f" Tỷ lệ cặp fake thuộc top real: {pair_metrics['fake_to_real_precision']:.3f}")


if __name__ == '__main__':
    args = get_generate_args()
    generate(args)
