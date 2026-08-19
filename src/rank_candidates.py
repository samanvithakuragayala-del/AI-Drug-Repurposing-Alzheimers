import pandas as pd
import requests
import time

# Load model predictions
df = pd.read_csv("model_test_predictions.csv")

# Rank by predicted probability
df = df.sort_values(
    "predicted_probability",
    ascending=False
).copy()

# Keep high-confidence predictions
top = df[
    df["predicted_probability"] >= 0.70
].copy()

# Remove duplicate drugs
top = top.drop_duplicates(
    subset=["molecule_chembl_id"]
)

# Keep the best 20 candidates
top = top.head(20).copy()

print("Top predicted candidates:")
print(
    top[
        [
            "molecule_chembl_id",
            "gene",
            "predicted_probability",
            "pchembl_value"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# Retrieve molecule information from ChEMBL
# --------------------------------------------------

names = []
max_phases = []

for chembl_id in top["molecule_chembl_id"]:

    url = (
        f"https://www.ebi.ac.uk/chembl/api/data/"
        f"molecule/{chembl_id}.json"
    )

    try:
        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        molecule = response.json()

        pref_name = molecule.get(
            "pref_name"
        )

        max_phase = molecule.get(
            "max_phase"
        )

        names.append(
            pref_name if pref_name else "Unknown"
        )

        max_phases.append(
            max_phase
        )

    except Exception:
        names.append("Unknown")
        max_phases.append(None)

    time.sleep(0.2)

top["drug_name"] = names
top["max_phase"] = max_phases

# Reorder columns
final = top[
    [
        "drug_name",
        "molecule_chembl_id",
        "gene",
        "predicted_probability",
        "pchembl_value",
        "max_phase"
    ]
]

# Save
final.to_csv(
    "top_candidates.csv",
    index=False
)

print("\n================================")
print("TOP CANDIDATES")
print("================================")

print(
    final.to_string(index=False)
)

print(
    "\nSaved: top_candidates.csv"
)