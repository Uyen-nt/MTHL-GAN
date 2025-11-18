import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json

def evaluate_dual_outputs(real_x, real_lens, fake_x, fake_lens, Vd, Vp):
    """
    Đánh giá riêng diagnoses và procedures
    """
    results = {}
    
    # 1. Phân tách diagnoses vs procedures
    real_diag = real_x[:, :, :Vd]
    real_proc = real_x[:, :, Vd:Vd+Vp]
    fake_diag = fake_x[:, :, :Vd]
    fake_proc = fake_x[:, :, Vd:Vd+Vp]
    
    # 2. Tính statistics cơ bản
    results['diag_stats'] = {
        'real_avg_codes_per_visit': real_diag.sum() / real_lens.sum(),
        'fake_avg_codes_per_visit': fake_diag.sum() / fake_lens.sum(),
        'real_visit_with_diag': (real_diag.sum(axis=-1) > 0).sum() / real_lens.sum(),
        'fake_visit_with_diag': (fake_diag.sum(axis=-1) > 0).sum() / fake_lens.sum()
    }
    
    results['proc_stats'] = {
        'real_avg_codes_per_visit': real_proc.sum() / real_lens.sum(),
        'fake_avg_codes_per_visit': fake_proc.sum() / fake_lens.sum(),
        'real_visit_with_proc': (real_proc.sum(axis=-1) > 0).sum() / real_lens.sum(),
        'fake_visit_with_proc': (fake_proc.sum(axis=-1) > 0).sum() / fake_lens.sum()
    }
    
    # 3. Tính visit có cả diag và proc (QUAN TRỌNG)
    real_joint = ((real_diag.sum(axis=-1) > 0) & (real_proc.sum(axis=-1) > 0)).sum() / real_lens.sum()
    fake_joint = ((fake_diag.sum(axis=-1) > 0) & (fake_proc.sum(axis=-1) > 0)).sum() / fake_lens.sum()
    
    results['joint_visits'] = {
        'real_ratio': real_joint,
        'fake_ratio': fake_joint,
        'ratio_difference': abs(real_joint - fake_joint)
    }
    
    return results

def analyze_visit_distribution(fake_x, fake_lens, Vd, Vp, sample_size=100):
    """
    Phân tích chi tiết distribution của visits
    """
    print("🔍 PHÂN TÍCH VISIT DISTRIBUTION:")
    print("=" * 50)
    
    for pid in range(min(sample_size, len(fake_x))):
        num_visits = fake_lens[pid]
        print(f"\nBệnh nhân {pid} ({num_visits} visits):")
        
        for vid in range(num_visits):
            visit = fake_x[pid, vid]
            diag_codes = np.where(visit[:Vd] > 0)[0]
            proc_codes = np.where(visit[Vd:Vd+Vp] > 0)[0]
            
            diag_count = len(diag_codes)
            proc_count = len(proc_codes)
            has_both = diag_count > 0 and proc_count > 0
            
            status = "✅ CẢ HAI" if has_both else "❌ CHỈ 1 LOẠI"
            
            print(f"  Visit {vid+1}: {status}")
            print(f"    Diagnoses: {diag_count} codes {list(diag_codes)[:5]}{'...' if diag_count > 5 else ''}")
            print(f"    Procedures: {proc_count} codes {list(proc_codes)[:5]}{'...' if proc_count > 5 else ''}")

def calculate_co_occurrence_metrics(real_x, fake_x, Vd, Vp):
    """
    Tính metrics về mối quan hệ diagnoses-procedures
    """
    def _calculate_cooccurrence_matrix(data, Vd, Vp):
        """Tính ma trận đồng xuất hiện diag-proc"""
        cooccurrence = np.zeros((Vd, Vp))
        count = 0
        
        for patient in data:
            for visit in patient:
                diag_indices = np.where(visit[:Vd] > 0)[0]
                proc_indices = np.where(visit[Vd:Vd+Vp] > 0)[0]
                
                if len(diag_indices) > 0 and len(proc_indices) > 0:
                    count += 1
                    for diag in diag_indices:
                        for proc in proc_indices:
                            cooccurrence[diag, proc] += 1
        
        return cooccurrence, count
    
    real_cooccur, real_count = _calculate_cooccurrence_matrix(real_x, Vd, Vp)
    fake_cooccur, fake_count = _calculate_cooccurrence_matrix(fake_x, Vd, Vp)
    
    # Tính similarity giữa 2 ma trận
    from scipy.spatial.distance import jensenshannon
    real_flat = real_cooccur.flatten() / (real_count + 1e-8)
    fake_flat = fake_cooccur.flatten() / (fake_count + 1e-8)
    
    js_distance = jensenshannon(real_flat, fake_flat)
    
    return {
        'real_cooccurring_visits': real_count,
        'fake_cooccurring_visits': fake_count,
        'cooccurrence_js_distance': js_distance,
        'real_cooccurrence_density': real_count / (len(real_x) * real_x.shape[1]),
        'fake_cooccurrence_density': fake_count / (len(fake_x) * fake_x.shape[1])
    }

# ======================================================
# THÊM MỚI: CÁC METRIC NÂNG CAO CHO DUAL OUTPUT
# ======================================================

def calculate_code_coverage_metrics(real_x, fake_x, Vd, Vp, rare_threshold=10):
    """
    Đánh giá:
    - Tổng số mã độc nhất được sinh ra (diagnoses + procedures riêng biệt)
    - Số mã hiếm (xuất hiện <= rare_threshold lần trong real) được sinh ra
    """
    def get_present_codes(data, Vd, Vp):
        diag_codes = set()
        proc_codes = set()
        
        for patient in data:
            for visit in patient:
                diag_active = np.where(visit[:Vd] > 0)[0]
                proc_active = np.where(visit[Vd:Vd+Vp] > 0)[0]
                
                diag_codes.update(diag_active)
                proc_codes.update(proc_active - Vd)  # chuyển về index gốc của proc
        
        return diag_codes, proc_codes
    
    # Real codes
    real_diag_codes, real_proc_codes = get_present_codes(real_x, Vd, Vp)
    
    # Fake codes
    fake_diag_codes, fake_proc_codes = get_present_codes(fake_x, Vd, Vp)
    
    # Tính tần suất trong real để xác định mã hiếm
    real_diag_freq = np.zeros(Vd)
    real_proc_freq = np.zeros(Vp)
    
    for patient in real_x:
        for visit in patient:
            real_diag_freq += (visit[:Vd] > 0)
            real_proc_freq += (visit[Vd:Vd+Vp] > 0)
    
    rare_diag_real = set(np.where(real_diag_freq <= rare_threshold)[0])
    rare_proc_real = set(np.where(real_proc_freq <= rare_threshold)[0])
    
    # Coverage
    diag_coverage = len(fake_diag_codes) / len(real_diag_codes) if real_diag_codes else 0
    proc_coverage = len(fake_proc_codes) / len(real_proc_codes) if real_proc_codes else 0
    
    # Rare code recall
    rare_diag_recall = len(fake_diag_codes & rare_diag_real) / len(rare_diag_real) if rare_diag_real else 0
    rare_proc_recall = len(fake_proc_codes & rare_proc_real) / len(rare_proc_real) if rare_proc_real else 0
    
    return {
        'diagnosis': {
            'real_unique': len(real_diag_codes),
            'fake_unique': len(fake_diag_codes),
            'coverage_ratio': diag_coverage,
            'rare_real': len(rare_diag_real),
            'rare_generated': len(fake_diag_codes & rare_diag_real),
            'rare_recall': rare_diag_recall
        },
        'procedure': {
            'real_unique': len(real_proc_codes),
            'fake_unique': len(fake_proc_codes),
            'coverage_ratio': proc_coverage,
            'rare_real': len(rare_proc_real),
            'rare_generated': len(fake_proc_codes & rare_proc_real),
            'rare_recall': rare_proc_recall
        },
        'rare_threshold': rare_threshold
    }


def calculate_pairwise_cooccurrence_precision_recall(real_x, fake_x, Vd, Vp, top_k_pairs=1000):
    """
    Tính Precision/Recall của các cặp (diagnosis, procedure) đồng xuất hiện
    Dựa trên các cặp phổ biến nhất trong dữ liệu thật
    """
    from collections import defaultdict
    
    def extract_cooccur_pairs(data, Vd, Vp):
        pairs = defaultdict(int)
        for patient in data:
            for visit in patient:
                diags = set(np.where(visit[:Vd] > 0)[0])
                procs = set(np.where(visit[Vd:Vd+Vp] > 0)[0]) - set(range(Vd))  # offset
                procs = {p - Vd for p in procs}
                
                for d in diags:
                    for p in procs:
                        pairs[(d, p)] += 1
        return pairs
    
    real_pairs_count = extract_cooccur_pairs(real_x, Vd, Vp)
    fake_pairs_count = extract_cooccur_pairs(fake_x, Vd, Vp)
    
    # Lấy top_k cặp phổ biến nhất trong real
    top_real_pairs = sorted(real_pairs_count.items(), key=lambda x: x[1], reverse=True)[:top_k_pairs]
    top_real_set = {(d, p) for ((d, p), count) in top_real_pairs}
    
    # Tính intersection với fake
    fake_set = set(fake_pairs_count.keys())
    intersection = top_real_set & fake_set
    
    precision = len(intersection) / len(top_real_set) if top_real_set else 0
    recall = len(intersection) / len(top_real_set) if top_real_set else 0  # recall = precision ở đây vì cùng mẫu tử
    
    # Bonus: tỷ lệ cặp fake nằm trong top real (đo "realistic co-occurrence")
    reverse_precision = len(intersection) / len(fake_set) if fake_set else 0
    
    return {
        'top_k': top_k_pairs,
        'real_top_pairs': len(top_real_pairs),
        'generated_matching_pairs': len(intersection),
        'cooccurrence_precision': precision,
        'cooccurrence_recall': recall,
        'fake_to_real_precision': reverse_precision,
        'total_fake_pairs': len(fake_set)
    }
