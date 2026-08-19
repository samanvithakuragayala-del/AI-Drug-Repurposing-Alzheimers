import requests
import pandas as pd
import time

# Load Alzheimer's-associated targets
targets = pd.read_csv("ad_targets.csv")

# Keep the strongest 20 targets for our first analysis
targets = targets.head(20)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

all_activities = []

for _, row in targets.iterrows():

    gene = row["gene"]

    print(f"\nSearching ChEMBL for target: {gene}")

    # Search ChEMBL targets
    search_url = f"{BASE_URL}/target/search.json"
    params = {"q": gene}

    response = requests.get(
        search_url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    results = response.json().get("targets", [])

    if not results:
        print(f"No ChEMBL target found for {gene}")
        continue

    # Find a human single-protein target when possible
    selected = None

    for target in results:
        if (
            target.get("organism") == "Homo sapiens"
            and target.get("target_type") == "SINGLE PROTEIN"
        ):
            selected = target
            break

    if selected is None:
        selected = results[0]

    chembl_target_id = selected["target_chembl_id"]

    print(
        f"Using: {chembl_target_id} - "
        f"{selected.get('pref_name', 'Unknown')}"
    )

    # Retrieve activities for this target
    activity_url = f"{BASE_URL}/activity.json"

    params = {
        "target_chembl_id": chembl_target_id,
        "limit": 1000
    }

    response = requests.get(
        activity_url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    activities = response.json().get("activities", [])

    for activity in activities:

        all_activities.append({
            "gene": gene,
            "chembl_target_id": chembl_target_id,
            "molecule_chembl_id": activity.get("molecule_chembl_id"),
            "canonical_smiles": activity.get("canonical_smiles"),
            "standard_type": activity.get("standard_type"),
            "standard_value": activity.get("standard_value"),
            "standard_units": activity.get("standard_units"),
            "pchembl_value": activity.get("pchembl_value")
        })

    time.sleep(0.5)


df = pd.DataFrame(all_activities)

print("\nTotal activity records:", len(df))

# Keep useful activity measurements
if not df.empty:

    df = df[
        df["standard_type"].isin(
            ["IC50", "Ki", "Kd", "EC50"]
        )
    ]

    df = df.dropna(
        subset=["molecule_chembl_id"]
    )

    df = df.drop_duplicates()

    df.to_csv(
        "chembl_ad_activities.csv",
        index=False
    )

    print(
        "\nSaved: chembl_ad_activities.csv"
    )

    print(
        "\nFinal dataset shape:",
        df.shape
    )

    print(
        "\nActivity types:"
    )

    print(
        df["standard_type"].value_counts()
    )

else:

    print("No activity data retrieved.")