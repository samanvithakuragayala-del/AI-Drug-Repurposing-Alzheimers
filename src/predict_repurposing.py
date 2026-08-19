import pandas as pd
import numpy as np
import requests

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("Loading ML dataset...")

df = pd.read_csv("ml_dataset.csv")

print("Dataset shape:", df.shape)


# ==========================================================
# 2. CREATE 2048-BIT MORGAN FINGERPRINTS
# ==========================================================

print("\nGenerating molecular fingerprints...")

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fingerprints = []
valid_rows = []

for index, smiles in df["canonical_smiles"].items():

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        continue

    fp = generator.GetFingerprintAsNumPy(mol)

    fingerprints.append(fp)
    valid_rows.append(index)


df = df.loc[valid_rows].reset_index(drop=True)

fingerprints = np.array(fingerprints)

print("Valid molecules:", len(df))
print("Fingerprint size:", fingerprints.shape[1])


# ==========================================================
# 3. ENCODE ALZHEIMER'S TARGETS
# ==========================================================

print("\nEncoding targets...")

target_names = sorted(
    df["gene"].dropna().unique()
)

target_features = np.zeros(
    (len(df), len(target_names)),
    dtype=int
)

target_index = {
    target: i
    for i, target in enumerate(target_names)
}

for i, target in enumerate(df["gene"]):

    if target in target_index:
        target_features[
            i,
            target_index[target]
        ] = 1


# ==========================================================
# 4. CREATE MODEL INPUT
# ==========================================================

X = np.hstack(
    [
        fingerprints,
        target_features
    ]
)

y = df["interaction_label"].astype(int)


print("\nModel input shape:", X.shape)

print("\nClass distribution:")
print(y.value_counts())


# ==========================================================
# 5. TRAIN RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X, y)

print("Model training completed.")


# ==========================================================
# 6. GET APPROVED DRUGS FROM CHEMBL
# ==========================================================

print("\nRetrieving approved drugs from ChEMBL...")

approved_drugs = []

url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

offset = 0
limit = 1000

while True:

    params = {
        "max_phase": 4,
        "limit": limit,
        "offset": offset
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    molecules = data.get(
        "molecules",
        []
    )

    if not molecules:
        break

    approved_drugs.extend(molecules)

    print(
        "Retrieved:",
        len(approved_drugs)
    )

    if len(molecules) < limit:
        break

    offset += limit


print(
    "\nTotal approved drug records:",
    len(approved_drugs)
)


# ==========================================================
# 7. EXTRACT DRUG INFORMATION
# ==========================================================

drug_rows = []

for molecule in approved_drugs:

    chembl_id = molecule.get(
        "molecule_chembl_id"
    )

    drug_name = molecule.get(
        "pref_name"
    )

    structures = molecule.get(
        "molecule_structures"
    )

    if not structures:
        continue

    smiles = structures.get(
        "canonical_smiles"
    )

    if not smiles:
        continue

    drug_rows.append(
        {
            "drug_name": drug_name,
            "molecule_chembl_id": chembl_id,
            "canonical_smiles": smiles
        }
    )


drugs = pd.DataFrame(
    drug_rows
)

drugs = drugs.drop_duplicates(
    subset="molecule_chembl_id"
)

print(
    "Approved drugs with structures:",
    len(drugs)
)


# ==========================================================
# 8. GENERATE DRUG FINGERPRINTS
# ==========================================================

print("\nGenerating fingerprints for approved drugs...")

drug_fingerprints = []
valid_drugs = []

for _, drug in drugs.iterrows():

    mol = Chem.MolFromSmiles(
        drug["canonical_smiles"]
    )

    if mol is None:
        continue

    fp = generator.GetFingerprintAsNumPy(
        mol
    )

    drug_fingerprints.append(fp)
    valid_drugs.append(drug)


drugs = pd.DataFrame(
    valid_drugs
).reset_index(drop=True)

drug_fingerprints = np.array(
    drug_fingerprints
)

print(
    "Valid approved drugs:",
    len(drugs)
)


# ==========================================================
# 9. GET ALZHEIMER'S TARGETS
# ==========================================================

targets = (
    df[
        [
            "gene",
            "chembl_target_id"
        ]
    ]
    .drop_duplicates()
)

print(
    "\nAlzheimer's targets:",
    len(targets)
)


# ==========================================================
# 10. FIND KNOWN INTERACTIONS
# ==========================================================

known_pairs = set(
    zip(
        df["molecule_chembl_id"],
        df["chembl_target_id"]
    )
)


# ==========================================================
# 11. GENERATE NEW DRUG-TARGET PAIRS
# ==========================================================

print(
    "\nGenerating new approved-drug / "
    "Alzheimer's-target combinations..."
)

candidate_rows = []

for drug_index, drug in drugs.iterrows():

    drug_fp = drug_fingerprints[
        drug_index
    ]

    for _, target in targets.iterrows():

        drug_id = drug[
            "molecule_chembl_id"
        ]

        target_id = target[
            "chembl_target_id"
        ]

        gene = target[
            "gene"
        ]

        # Skip interactions already present
        # in our ChEMBL dataset
        if (
            drug_id,
            target_id
        ) in known_pairs:
            continue

        # Target encoding
        target_vector = np.zeros(
            len(target_names)
        )

        if gene in target_index:

            target_vector[
                target_index[gene]
            ] = 1

        features = np.concatenate(
            [
                drug_fp,
                target_vector
            ]
        )

        candidate_rows.append(
            {
                "drug_name": drug[
                    "drug_name"
                ],
                "molecule_chembl_id": drug_id,
                "gene": gene,
                "chembl_target_id": target_id,
                "features": features
            }
        )


candidates = pd.DataFrame(
    candidate_rows
)

print(
    "Novel candidate pairs:",
    len(candidates)
)


# ==========================================================
# 12. PREDICT INTERACTION PROBABILITY
# ==========================================================

print(
    "\nPredicting interaction probabilities..."
)

X_candidates = np.array(
    candidates["features"].tolist()
)

probabilities = model.predict_proba(
    X_candidates
)[:, 1]

candidates[
    "predicted_probability"
] = probabilities


# ==========================================================
# 13. RANK CANDIDATES
# ==========================================================

candidates = candidates.sort_values(
    "predicted_probability",
    ascending=False
)


# Keep only high-probability predictions
high_confidence = candidates[
    candidates["predicted_probability"] >= 0.70
].copy()


# Keep the strongest prediction for
# each drug
high_confidence = (
    high_confidence
    .drop_duplicates(
        subset="molecule_chembl_id"
    )
    .head(20)
)


# ==========================================================
# 14. SAVE FINAL RESULTS
# ==========================================================

final_columns = [
    "drug_name",
    "molecule_chembl_id",
    "gene",
    "chembl_target_id",
    "predicted_probability"
]

final_results = high_confidence[
    final_columns
].copy()


final_results.to_csv(
    "novel_repurposing_candidates.csv",
    index=False
)


# ==========================================================
# 15. DISPLAY RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("TOP POTENTIAL REPURPOSING CANDIDATES")
print("=" * 60)

print(
    final_results.to_string(
        index=False
    )
)

print("\nSaved as:")
print("novel_repurposing_candidates.csv")

print("\nPROJECT STEP COMPLETED.")