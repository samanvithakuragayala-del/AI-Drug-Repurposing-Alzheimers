import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Load ChEMBL activity data
df = pd.read_csv("chembl_ad_activities.csv")

print("Original records:", len(df))

# Remove records without SMILES
df = df.dropna(subset=["canonical_smiles"]).copy()

print("Records with SMILES:", len(df))

# Remove duplicate drug-target combinations
df = df.drop_duplicates(
    subset=["molecule_chembl_id", "chembl_target_id"]
).copy()

print("Unique drug-target pairs:", len(df))

# Create Morgan fingerprint generator
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fingerprints = []
valid_indices = []

for index, smiles in df["canonical_smiles"].items():

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is not None:
        fingerprint = generator.GetFingerprintAsNumPy(molecule)
        fingerprints.append(fingerprint)
        valid_indices.append(index)

# Keep only valid molecules
df = df.loc[valid_indices].copy()

# Convert fingerprints to DataFrame
fp_array = np.array(fingerprints)

fp_columns = [
    f"FP_{i}"
    for i in range(fp_array.shape[1])
]

fp_df = pd.DataFrame(
    fp_array,
    columns=fp_columns,
    index=df.index
)

# Combine original data with fingerprints
final_df = pd.concat(
    [
        df,
        fp_df
    ],
    axis=1
)

# Save
final_df.to_csv(
    "ad_ml_features.csv",
    index=False
)

print("\nFEATURE GENERATION COMPLETE")
print("Final records:", len(final_df))
print("Fingerprint features:", len(fp_columns))
print("Final dataset shape:", final_df.shape)
print("\nSaved as: ad_ml_features.csv")