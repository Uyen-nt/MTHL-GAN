import os
import json
import numpy as np

def export_jsonl(hier_npz_path, diag_vocab_json=None, proc_vocab_json=None, out_path=None, top_k_visit=None):
    """
    Chuyển dữ liệu từ synthetic_mimic3_hier.npz thành cấu trúc JSONL:
    bệnh nhân -> lượt khám -> mã bệnh -> thủ thuật
    """

    print(f"📂 Loading: {hier_npz_path}")
    data = np.load(hier_npz_path)
    diag, proc, lens = data["diag"], data["proc"], data["lens"]
    data.close()

    n_patients, n_visits, n_diag = diag.shape
    n_proc = proc.shape[-1]
    print(f"✅ Loaded synthetic hierarchical data: {n_patients} patients, {n_visits} visits")

    # --- Load vocab ---
    diag_vocab = None
    proc_vocab = None
    if diag_vocab_json and os.path.exists(diag_vocab_json):
        diag_vocab = json.load(open(diag_vocab_json))
        diag_vocab = {int(v): k for k, v in diag_vocab.items()}  # idx->code
    if proc_vocab_json and os.path.exists(proc_vocab_json):
        proc_vocab = json.load(open(proc_vocab_json))
        proc_vocab = {int(v): k for k, v in proc_vocab.items()}  # idx->code

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
                # ✅ Lấy các mã có giá trị > 0 (sửa lỗi i,t)
                diag_codes = np.where(diag[pid, vid] > 0)[0].tolist()
                proc_codes = np.where(proc[pid, vid] > 0.05)[0].tolist()

                # Map index -> mã thật (nếu có vocab)
                diag_codes = [diag_vocab[i] if diag_vocab and i in diag_vocab else f"DIAG_{i}" for i in diag_codes]
                proc_codes = [proc_vocab[i] if proc_vocab and i in proc_vocab else f"PROC_{i}" for i in proc_codes]

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
