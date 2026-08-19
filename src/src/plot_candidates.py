import pandas as pd
import matplotlib.pyplot as plt

# Load filtered candidates
df = pd.read_csv("filtered_repurposing_candidates.csv")

# Standardize drug names
name_map = {
    "BENAZEPRIL HYDROCHLORIDE": "Benazepril",
    "BENAZEPRIL": "Benazepril",
    "SERTRALINE HYDROCHLORIDE": "Sertraline",
    "SERTRALINE": "Sertraline",
    "ENALAPRILAT": "Enalaprilat",
    "ENALAPRILAT ANHYDROUS": "Enalaprilat",
    "LISINOPRIL": "Lisinopril",
    "LISINOPRIL ANHYDROUS": "Lisinopril",
}

df["display_drug"] = (
    df["drug_name"]
    .str.upper()
    .map(name_map)
    .fillna(df["drug_name"].str.title())
)

# Keep the highest prediction for each distinct drug
df = (
    df.sort_values(
        "predicted_probability",
        ascending=False
    )
    .drop_duplicates(
        subset="display_drug"
    )
    .head(6)
)

# Sort for plotting
df = df.sort_values(
    "predicted_probability"
)

# Create figure
plt.figure(figsize=(9, 6))

plt.barh(
    df["display_drug"],
    df["predicted_probability"]
)

plt.xlabel("Predicted interaction probability")
plt.ylabel("Candidate drug")
plt.title(
    "AI-Prioritized Drug Repurposing Candidates"
)

plt.xlim(0, 1.05)

# Add values
for i, value in enumerate(
    df["predicted_probability"]
):
    plt.text(
        value + 0.01,
        i,
        f"{value:.3f}",
        va="center"
    )

plt.tight_layout()

plt.savefig(
    "top_repurposing_candidates.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Save clean table
df[
    [
        "display_drug",
        "gene",
        "predicted_probability"
    ]
].to_csv(
    "final_candidate_results.csv",
    index=False
)

print("\nFINAL CANDIDATES")
print(
    df[
        [
            "display_drug",
            "gene",
            "predicted_probability"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print("top_repurposing_candidates.png")
print("final_candidate_results.csv")