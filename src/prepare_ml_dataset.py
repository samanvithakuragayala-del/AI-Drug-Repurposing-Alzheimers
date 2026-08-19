import pandas as pd
import numpy as np

# Load feature dataset
df = pd.read_csv("ad_ml_features.csv")

print("Original rows:", len(df))

# Keep records with pChEMBL values
df = df.dropna(subset=["pchembl_value"]).copy()

# Convert to numeric
df["pchembl_value"] = pd.to_numeric(
    df["pchembl_value"],
    errors="coerce"
)

df = df.dropna(subset=["pchembl_value"])

# Keep only clearly active or clearly weak interactions
df["interaction_label"] = np.where(
    df["pchembl_value"] >= 6,
    1,
    np.where(
        df["pchembl_value"] <= 5,
        0,
        np.nan
    )
)

# Remove ambiguous measurements
df = df.dropna(
    subset=["interaction_label"]
).copy()

df["interaction_label"] = df[
    "interaction_label"
].astype(int)

# If multiple measurements exist for the same
# drug-target pair, retain the strongest observed activity
df = (
    df.groupby(
        ["molecule_chembl_id", "chembl_target_id"],
        as_index=False
    )
    .agg({
        "pchembl_value": "max",
        "interaction_label": "max",
        "canonical_smiles": "first",
        "gene": "first"
    })
)

print("Unique drug-target pairs:", len(df))

print("\nClass distribution:")
print(df["interaction_label"].value_counts())

# One-hot encode target/gene
target_encoded = pd.get_dummies(
    df["gene"],
    prefix="TARGET",
    dtype=int
)

# Identify molecular fingerprint columns
fingerprint_columns = [
    col for col in df.columns
    if col.startswith("FP_")
]

# Note: fingerprints were not retained by the groupby above,
# so reload them from the original feature file.
features = pd.read_csv("ad_ml_features.csv")

features = features[
    [
        "molecule_chembl_id",
        "chembl_target_id"
    ] + fingerprint_columns
].drop_duplicates()

# Merge fingerprints back
df = df.merge(
    features,
    on=["molecule_chembl_id", "chembl_target_id"],
    how="inner"
)

# Recreate target encoding after merge
target_encoded = pd.get_dummies(
    df["gene"],
    prefix="TARGET",
    dtype=int
)

# Combine
ml_df = pd.concat(
    [
        df.reset_index(drop=True),
        target_encoded.reset_index(drop=True)
    ],
    axis=1
)

# Save
ml_df.to_csv(
    "ml_dataset.csv",
    index=False
)

print("\nML DATASET READY")
print("Shape:", ml_df.shape)
print("Fingerprint features:", len(fingerprint_columns))
print("Target features:", len(target_encoded.columns))
print("\nSaved as: ml_dataset.csv")