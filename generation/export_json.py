import os
import json
import numpy as np

def export_jsonl(hier_npz_path, diag_vocab_json=None, proc_vocab_json=None, out_path=None, top_k_visit=None):
    """
    Chuyển dữ liệu từ unified synthetic_mimic3_hier.npz (x, lens)
    thành cấu trúc JSONL:
        bệnh nhân -> lượt khám -> mã bệnh + mã thủ thuật
    """

    print(f"📂 Loading unified hierarchical file: {hier_npz_path}")
    data = np.load(hier_npz_path)
    x, lens = data["x"], data["lens"]
    data.close()

    n_patients, n_visits, vocab_size = x.shape
    print(f"✅ Loaded synthetic unified data: {n_patients} patients, {n_visits} visits, vocab={vocab_size}")

    # --- Load hierarchical meta để biết Vd, Vp ---
    meta_path = os.path.join(os.path.dirname(hier_npz_path), "../data/mimic3/standard_hier/hier_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join("data/mimic3/standard_hier/hier_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("⚠️ hier_meta.json not found — cần để xác định Vd và Vp.")

    with open(meta_path) as f:
        meta = json.load(f)
    Vd, Vp = meta["Vd"], meta["Vp"]
    print(f"🔍 Loaded hierarchical meta: Vd={Vd}, Vp={Vp}, total={Vd+Vp}")

    # --- Load vocab (nếu có) ---
    diag_vocab = None
    proc_vocab = None
    if diag_vocab_json and os.path.exists(diag_vocab_json):
        diag_vocab = json.load(open(diag_vocab_json))
        diag_vocab = {int(v): k for k, v in diag_vocab.items()}
    if proc_vocab_json and os.path.exists(proc_vocab_json):
        proc_vocab = json.load(open(proc_vocab_json))
        proc_vocab = {int(v): k for k, v in proc_vocab.items()}

    # --- Output path ---
    if out_path is None:
        out_path = os.path.join(os.path.dirname(hier_npz_path), "fake_cases_with_procs.jsonl")

    print(f"💾 Writing to {out_path} ...")
    with open(out_path, "w", encoding="utf8") as f:
        for pid in range(n_patients):
            if top_k_visit and pid >= top_k_visit:
                break

            visits = []
            for vid in range(int(lens[pid])):
                visit_vec = x[pid, vid]

                # ✳️ Tách vùng bệnh & thủ thuật theo index
                diag_indices = np.where(visit_vec[:Vd] > 0.001)[0].tolist()
                proc_indices = np.where(visit_vec[Vd:] > 0.001)[0].tolist()

                # ✳️ Map index -> mã thật
                diag_codes = [
                    diag_vocab[i] if diag_vocab and i in diag_vocab else f"DIAG_{i}"
                    for i in diag_indices
                ]
                proc_codes = [
                    proc_vocab[i] if proc_vocab and i in proc_vocab else f"PROC_{i}"
                    for i in proc_indices
                ]

                visits.append({
                    "visit_id": vid + 1,
                    "diagnoses": diag_codes,
                    "procedures": proc_codes
                })

            case = {
                "case_id": f"fake_{pid:06d}",
                "visits": visits
            }
            json.dump(case, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Done! Exported {n_patients} fake cases.")
    return out_path
