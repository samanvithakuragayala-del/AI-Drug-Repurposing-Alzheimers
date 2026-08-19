import pandas as pd

# Load AI predictions
df = pd.read_csv("novel_repurposing_candidates.csv")

print("Original candidates:", len(df))

# Drugs that should NOT be considered therapeutic
# repurposing candidates for this project
exclude_terms = [
    "FLORBETABEN",
    "FLORBETAPIR",
    "FLUTEMETAMOL",
    "IOFLUPANE",
    "TACRINE"
]

# Remove diagnostic radiotracers and existing AD drug
pattern = "|".join(exclude_terms)

filtered = df[
    ~df["drug_name"]
    .fillna("")
    .str.upper()
    .str.contains(pattern, regex=True)
].copy()

# Remove duplicate active drug names where possible
filtered = filtered.drop_duplicates(
    subset=["drug_name"]
)

# Rank by AI probability
filtered = filtered.sort_values(
    "predicted_probability",
    ascending=False
)

# Keep top 10
filtered = filtered.head(10)

# Save
filtered.to_csv(
    "filtered_repurposing_candidates.csv",
    index=False
)

print("\n========================================")
print("FILTERED REPURPOSING CANDIDATES")
print("========================================")

print(
    filtered.to_string(index=False)
)

print("\nSaved as:")
print("filtered_repurposing_candidates.csv")