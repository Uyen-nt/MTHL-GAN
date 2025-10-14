from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_concept(patient_admission, admission_concepts):
    concept_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission[EHRParser.adm_id_col]
            if adm_id in admission_concepts:
                concepts = admission_concepts[adm_id]
                for concept in concepts:
                    if concept not in concept_map:
                        concept_map[concept] = len(concept_map)

    admission_concept_encoded = {
        admission_id: list(set(concept_map[concept] for concept in concept))
        for admission_id, concept in admission_concepts.items()
    }
    return admission_concept_encoded, concept_map

def encode_dual_concept(patient_admission, admission_diagnoses, admission_procedures):
    """
    Encode cả bệnh (diagnoses) và thủ thuật (procedures) vào cùng vocab.
    Diagnoses nằm trước, procedures nằm sau.
    """
    from collections import OrderedDict
    concept_map = OrderedDict()

    # ----- encode DIAGNOSES trước -----
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission['adm_id']
            if adm_id in admission_diagnoses:
                for code in admission_diagnoses[adm_id]:
                    if code not in concept_map:
                        concept_map[code] = len(concept_map)

    Vd = len(concept_map)

    # ----- encode PROCEDURES sau -----
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission['adm_id']
            if adm_id in admission_procedures:
                for code in admission_procedures[adm_id]:
                    if code not in concept_map:
                        concept_map[code] = len(concept_map)

    Vp = len(concept_map) - Vd
    print(f"[encode_dual_concept] Vd={Vd}, Vp={Vp}, total V={len(concept_map)}")

    # ----- map admission -> indices -----
    admission_concept_encoded = {}
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission['adm_id']
            diag_codes = admission_diagnoses.get(adm_id, [])
            proc_codes = admission_procedures.get(adm_id, [])
            all_codes = list(set(diag_codes + proc_codes))
            admission_concept_encoded[adm_id] = [concept_map[c] for c in all_codes if c in concept_map]

    return admission_concept_encoded, concept_map, Vd, Vp

