import requests
import pandas as pd

# Alzheimer's disease in MONDO
DISEASE_ID = "MONDO_0004975"

url = "https://api.platform.opentargets.org/api/v4/graphql"

query = """
query DiseaseTargets($diseaseId: String!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: {index: 0, size: 100}) {
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""

variables = {
    "diseaseId": DISEASE_ID
}

response = requests.post(
    url,
    json={"query": query, "variables": variables},
    timeout=60
)

response.raise_for_status()

data = response.json()

if "errors" in data:
    print("API ERROR:")
    print(data["errors"])
    raise SystemExit

disease = data["data"]["disease"]

rows = []

for item in disease["associatedTargets"]["rows"]:
    target = item["target"]

    rows.append({
        "target_id": target["id"],
        "gene": target["approvedSymbol"],
        "target_name": target["approvedName"],
        "association_score": item["score"]
    })

df = pd.DataFrame(rows)

df = df.sort_values(
    "association_score",
    ascending=False
)

df.to_csv(
    "ad_targets.csv",
    index=False
)

print("\nAlzheimer's disease:", disease["name"])
print("Number of targets retrieved:", len(df))
print("\nTop 20 targets:")
print(df.head(20).to_string(index=False))
print("\nSaved as: ad_targets.csv")