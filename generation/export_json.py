import os
import json
import numpy as np

def export_json(hier_npz_path, diag_vocab_json=None, proc_vocab_json=None, out_path=None, top_k_patients=None):
    """
    Chuyển dữ liệu từ unified synthetic_mimic3_hier.npz (x, lens)
    thành cấu trúc JSONL: HIỂN THỊ TẤT CẢ VISITS
    """
    print(f"📂 Loading unified hierarchical file: {hier_npz_path}")
    data = np.load(hier_npz_path)
    x, lens = data["x"], data["lens"]
    data.close()

    n_patients, n_visits, vocab_size = x.shape
    print(f"✅ Loaded synthetic unified data: {n_patients} patients, {n_visits} visits, vocab={vocab_size}")

    # Load hierarchical meta
    meta_path = os.path.join(os.path.dirname(hier_npz_path), "../data/mimic3/standard_hier/hier_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join("data/mimic3/standard_hier/hier_meta.json")
    
    with open(meta_path) as f:
        meta = json.load(f)
    Vd, Vp = meta["Vd"], meta["Vp"]
    print(f"🔍 Loaded hierarchical meta: Vd={Vd}, Vp={Vp}, total={Vd+Vp}")

    # Load vocab
    diag_vocab = None
    proc_vocab = None
    if diag_vocab_json and os.path.exists(diag_vocab_json):
        diag_vocab = json.load(open(diag_vocab_json))
        diag_vocab = {int(v): k for k, v in diag_vocab.items()}
    if proc_vocab_json and os.path.exists(proc_vocab_json):
        proc_vocab = json.load(open(proc_vocab_json))
        proc_vocab = {int(v): k for k, v in proc_vocab.items()}

    # Output path
    if out_path is None:
        out_path = os.path.join(os.path.dirname(hier_npz_path), "fake_cases_complete.jsonl")

    print(f"💾 Writing to {out_path} ...")
    
    stats = {
        'total_patients': 0,
        'total_visits': 0,
        'visits_with_both': 0,
        'visits_with_diag_only': 0,
        'visits_with_proc_only': 0,
        'visits_empty': 0
    }
    
    with open(out_path, "w", encoding="utf8") as f:
        for pid in range(n_patients):
            if top_k_patients and pid >= top_k_patients:
                break

            visits = []
            num_visits = int(lens[pid])
            
            for vid in range(num_visits):
                visit_vec = x[pid, vid]
                
                # Tách diagnoses và procedures
                diag_indices = np.where(visit_vec[:Vd] > 0.001)[0].tolist()
                proc_indices = np.where(visit_vec[Vd:Vd+Vp] > 0.001)[0].tolist()
                
                # Map to actual codes
                diag_codes = [
                    diag_vocab[i] if diag_vocab and i in diag_vocab else f"DIAG_{i}"
                    for i in diag_indices
                ]
                proc_codes = [
                    proc_vocab[i] if proc_vocab and i in proc_vocab else f"PROC_{i}"
                    for i in proc_indices
                ]
                
                # Thống kê
                stats['total_visits'] += 1
                if diag_codes and proc_codes:
                    stats['visits_with_both'] += 1
                elif diag_codes:
                    stats['visits_with_diag_only'] += 1
                elif proc_codes:
                    stats['visits_with_proc_only'] += 1
                else:
                    stats['visits_empty'] += 1

                visits.append({
                    "visit_id": vid + 1,
                    "diagnoses": diag_codes,
                    "procedures": proc_codes,
                    "diagnosis_count": len(diag_codes),
                    "procedure_count": len(proc_codes),
                    "has_both": len(diag_codes) > 0 and len(proc_codes) > 0
                })

            case = {
                "case_id": f"fake_{pid:06d}",
                "total_visits": num_visits,
                "visits": visits
            }
            json.dump(case, f, ensure_ascii=False)
            f.write("\n")
            stats['total_patients'] += 1

    # Print statistics
    print("\n📊 THỐNG KÊ KẾT QUẢ:")
    print(f"Tổng bệnh nhân: {stats['total_patients']}")
    print(f"Tổng visits: {stats['total_visits']}")
    print(f"Visits có cả diag + proc: {stats['visits_with_both']} ({stats['visits_with_both']/stats['total_visits']*100:.1f}%)")
    print(f"Visits chỉ có diag: {stats['visits_with_diag_only']} ({stats['visits_with_diag_only']/stats['total_visits']*100:.1f}%)")
    print(f"Visits chỉ có proc: {stats['visits_with_proc_only']} ({stats['visits_with_proc_only']/stats['total_visits']*100:.1f}%)")
    print(f"Visits trống: {stats['visits_empty']} ({stats['visits_empty']/stats['total_visits']*100:.1f}%)")

    return out_path
