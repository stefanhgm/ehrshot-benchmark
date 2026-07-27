# ehrshot/helper/code_category_statistics.py
import argparse
import os
import sys
from collections import Counter

from tqdm import tqdm

# --- bootstrap imports for both `ehrshot.*` AND legacy `serialization.*` ---
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_EHRSHOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # .../ehrshot

for p in (_REPO_ROOT, _EHRSHOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
# -------------------------------------------------------------------------

from femr.extension import datasets as extension_datasets

# reuse existing resolve_code logic
from ehrshot.llm_featurizer import LLMFeaturizer
from ehrshot.serialization.ehr_simple_serializer import EHRSimpleSerializer, UniqueCodesListRecentStrategy

def _excluded_onts(name: str):
    return (
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3'] if name == 'no_labs' else
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'RxNorm', 'RxNorm Extension'] if name == 'no_labs_meds' else
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'Medicare Specialty', 'CMS Place of Service', 'OMOP Extension', 'Condition Type'] if name == 'no_labs_single' else
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'RxNorm', 'RxNorm Extension', 'Medicare Specialty', 'CMS Place of Service', 'OMOP Extension', 'Condition Type'] if name == 'no_labs_meds_single' else
        ['CARE_SITE', 'ICDO3'] if name == 'no_unres' else
        []
    )


def main(path_to_database: str, excluded_ontologies: str) -> None:
    db = extension_datasets.PatientDatabase(path_to_database)
    ontology = db.get_ontology()

    # Reuse existing resolve_code logic from LLMFeaturizer
    featurizer = LLMFeaturizer(
        embedding_size=0,
        serialization_strategy=UniqueCodesListRecentStrategy(0),
        excluded_ontologies=_excluded_onts(excluded_ontologies),
    )

    ser = EHRSimpleSerializer(ontology=ontology)  # needed for SNOMED parents in classify()

    counts, total = Counter(), 0
    pids = list(db.patient_ids()) if hasattr(db, "patient_ids") else list(db.keys())  # type: ignore

    resolve_code = lambda code: featurizer.resolve_code_with_custom_ontologies(ontology, code)

    for pid in tqdm(pids, desc="Patients"):
        patient = db[pid]  # type: ignore
        ser.load_from_femr_events(patient.events, resolve_code, is_visit_event=lambda _: False, filter_aggregated_events=False)
        for e in ser.static_events:
            if e.code:
                counts[ser.classify(e.code)] += 1
                total += 1

    order = ["demographics", "visits", "medications", "procedures", "labs", "conditions"]
    print(f"Total codes: {total}")
    for cat in order + sorted([c for c in counts if c not in order]):
        n = counts.get(cat, 0)
        pct = 0.0 if total == 0 else 100.0 * n / total
        print(f"{cat:13s} {n:10d}  {pct:5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_to_database", required=True)
    ap.add_argument("--excluded_ontologies", default="no_unres")  # match common pipeline default
    args = ap.parse_args()
    main(args.path_to_database, args.excluded_ontologies)