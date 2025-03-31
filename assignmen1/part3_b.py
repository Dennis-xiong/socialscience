import requests
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import time

BASE_URL = "https://api.openalex.org/works"
EMAIL = "hangyuxiong9@gmail.com"


def fetch_works(author_ids, per_page=200):
    works_data = []
    abstracts_data = []
    page = 1
    all_concepts = "C15744967|C138885662|C71924100|C142362112|C33923547|C121332964|C41008148"
    filters = (
        f"author.id:{'|'.join(author_ids)},"
        "cited_by_count:>10,"
        "authors_count:<10,"
        f"concepts.id:{all_concepts}"
    )
    headers = {"User-Agent": "Python-Research-Script/1.0"}

    while True:
        url = f"{BASE_URL}?filter={filters}&per-page={per_page}&page={page}&mailto={EMAIL}"
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"Page {page} returned {len(results)} results")
            if not results:
                break
            for work in results:
                # Directly save all data that matches the API filter criteria
                works_data.append({
                    "id": work["id"],
                    "publication_year": work["publication_year"],
                    "cited_by_count": work["cited_by_count"],
                    "author_ids": [auth["author"]["id"] for auth in work["authorships"]]
                })
                abstracts_data.append({
                    "id": work["id"],
                    "title": work["title"],
                    "abstract_inverted_index": work.get("abstract_inverted_index", {})
                })
            page += 1
        elif response.status_code == 429:
            print(f"Rate limit hit on page {page}, waiting 5 seconds...")
            time.sleep(5)
        else:
            print(f"Error fetching page {page}: {response.status_code} - {response.text}")
            break
        time.sleep(1)

    return works_data, abstracts_data


def process_authors_chunk(author_chunk):
    return fetch_works([author["id"] for author in author_chunk])


# Load data
authors_df = pd.read_csv("IC2S2_2024_Computational_Social_Scientists.csv")
filtered_authors = authors_df[
    (authors_df["works_count"] >= 5) & (authors_df["works_count"] <= 5000)
    ].to_dict("records")

# Process in chunks
author_chunks = [filtered_authors[i:i + 25] for i in range(0, len(filtered_authors), 25)]

# Use a small number of parallel processes
results = Parallel(n_jobs=2)(delayed(process_authors_chunk)(chunk) for chunk in tqdm(author_chunks))

# Combine results
all_works = []
all_abstracts = []
for works, abstracts in results:
    all_works.extend(works)
    all_abstracts.extend(abstracts)

# Create DataFrame
papers_df = pd.DataFrame(all_works)
abstracts_df = pd.DataFrame(all_abstracts)

# Save files
if papers_df.empty:
    print("Warning: No data retrieved, saving empty files.")
papers_df.to_csv("IC2S2_papers.csv", index=False)
abstracts_df.to_csv("IC2S2_abstracts.csv", index=False)


# Data overview
print(f"Number of works in IC2S2 papers: {len(papers_df)}")
if not papers_df.empty:
    unique_authors = len(set().union(*papers_df["author_ids"]))
    print(f"Number of unique researchers: {unique_authors}")
else:
    print("No data retrieved, cannot compute unique researchers.")
print(authors_df.columns)

