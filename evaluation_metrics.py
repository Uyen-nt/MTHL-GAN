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
