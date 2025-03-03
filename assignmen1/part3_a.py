import requests
import pandas as pd
from difflib import SequenceMatcher
from joblib import Parallel, delayed
import time

def similar(a, b):
    """Return the similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def process_name(name):
    search_url = f"https://api.openalex.org/authors?search={name}"
    try:
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        results = response.json().get("results", [])
        best_match = None
        best_similarity = 0.0
        for author in results:
            sim = similar(name, author.get("display_name", ""))
            if sim > best_similarity:
                best_similarity = sim
                best_match = author
        if best_match and best_similarity >= 0.8:
            full_id = best_match.get("id")
            if not full_id.startswith("https://openalex.org/"):
                full_id = "https://openalex.org/" + full_id
            works_api_url = best_match.get("works_api_url")
            if works_api_url and not works_api_url.startswith("https://"):
                works_api_url = "https://" + works_api_url
            # Note here we use last_known_institutions (plural form)
            inst_list = best_match.get("last_known_institutions", [])
            if inst_list and isinstance(inst_list, list) and len(inst_list) > 0:
                country = inst_list[0].get("country_code", "Unknown")
            else:
                country = "Unknown"
            # h_index is now stored in summary_stats
            h_index = best_match.get("summary_stats", {}).get("h_index")
            return {
                "id": full_id,
                "display_name": best_match.get("display_name"),
                "works_api_url": works_api_url,
                "h_index": h_index,
                "works_count": best_match.get("works_count"),
                "country--code": country
            }
    except Exception:
        pass
    return None

# Read researcher names from a file
with open("IC2S2_2023_researchers.txt", "r", encoding="utf-8") as f:
    researcher_names = [line.strip() for line in f if line.strip()]

# Parallel processing (n_jobs can be adjusted as needed)
results = Parallel(n_jobs=5)(delayed(process_name)(name) for name in researcher_names)
data_list = [r for r in results if r is not None]

# Convert to DataFrame and save to CSV with prefix in column names
df = pd.DataFrame(data_list)
df.to_csv("IC2S2_2024_Computational_Social_Scientists.csv", index=False)
print("Data has been saved to IC2S2_2024_Computational_Social_Scientists.csv")
