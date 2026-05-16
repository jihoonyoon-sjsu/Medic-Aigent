# shared code
# need to update differently for notebooks

from pathlib import Path
import pandas as pd

DATA_DIR = Path("mimic_data/baseline_tables")
PROCESSED_DIR = Path("mimic_data/processed")
REFERENCE_DIR = Path("mimic_data/reference")

RANDOM_STATE = 42

# lab sources tables
LAB_SOURCES = [
    "chemistry",
    "complete_blood_count",
    "coagulation",
    "enzyme",
    "blood_differential",
    "cardiac_marker",
    "bg",
]


# raw tables
def load_raw_tables():
    admissions = pd.read_csv(DATA_DIR / "admissions.csv", low_memory=False)
    patients = pd.read_csv(DATA_DIR / "patients.csv", low_memory=False)
    diagnoses = pd.read_csv(DATA_DIR / "diagnoses_icd.csv", low_memory=False)
    emar = pd.read_csv(DATA_DIR / "emar.csv", low_memory=False)
    return admissions, patients, diagnoses, emar
