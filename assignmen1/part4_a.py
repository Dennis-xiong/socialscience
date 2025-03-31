import pandas as pd
import networkx as nx
import json
import ast
from itertools import combinations
from networkx.readwrite import json_graph

# -------------------------
# 1. Load Author Data
# -------------------------
# Assuming the author CSV contains columns: "id:", "display_name:", "works_api_url:",
# "h_index:", "works_count:", "country--code:"
authors_df = pd.read_csv("IC2S2_2024_Computational_Social_Scientists.csv")
# Remove trailing colons from column names
authors_df.columns = [col.rstrip(":") for col in authors_df.columns]

# -------------------------
# 2. Load Paper Data
# -------------------------
# Assuming the paper CSV has an "author_ids" column storing author ID lists as strings,
# e.g.: '["https://openalex.org/A123", "https://openalex.org/A456"]'
papers_df = pd.read_csv("IC2S2_papers.csv")

def parse_author_ids(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

papers_df["author_ids_parsed"] = papers_df["author_ids"].apply(parse_author_ids)

# -------------------------
# 3. Build Weighted Edge List
# -------------------------
# Process papers to generate author pairs (unordered combinations),
# count co-authored papers as edge weights
edge_dict = {}
for idx, row in papers_df.iterrows():
    authors = row["author_ids_parsed"]
    if len(authors) > 1:
        for pair in combinations(authors, 2):
            # Sort to ensure (A, B) and (B, A) are treated as the same pair
            sorted_pair = tuple(sorted(pair))
            edge_dict[sorted_pair] = edge_dict.get(sorted_pair, 0) + 1

weighted_edgelist = [(u, v, weight) for (u, v), weight in edge_dict.items()]

# -------------------------
# 4. Build Network & Add Node Attributes
# -------------------------
G = nx.Graph()

# 4.1 Add all author nodes (including isolated ones)
for idx, row in authors_df.iterrows():
    author_id = row["id"]  # Contains full prefix
    G.add_node(author_id)
    G.nodes[author_id]["display_name"] = row["display_name"]
    G.nodes[author_id]["works_api_url"] = row["works_api_url"]
    G.nodes[author_id]["h_index"] = row["h_index"]
    G.nodes[author_id]["works_count"] = row["works_count"]
    G.nodes[author_id]["country"] = row["country--code"]

# 4.2 Add collaboration edges
G.add_weighted_edges_from(weighted_edgelist)

# -------------------------
# 5. Add Paper-derived Node Attributes
# -------------------------
# Calculate total citations and earliest publication year per author
author_stats = {}
for idx, row in papers_df.iterrows():
    authors = row["author_ids_parsed"]
    pub_year = row.get("publication_year")
    cited = row.get("cited_by_count", 0)
    for author in authors:
        if author not in author_stats:
            author_stats[author] = {"years": [], "citations": 0}
        if pd.notnull(pub_year):
            author_stats[author]["years"].append(pub_year)
        author_stats[author]["citations"] += cited

for author, stats in author_stats.items():
    if author in G.nodes:
        G.nodes[author]["total_citations"] = stats["citations"]
        if stats["years"]:
            G.nodes[author]["first_publication_year"] = min(stats["years"])
        else:
            G.nodes[author]["first_publication_year"] = None

# Set defaults for authors without paper data
for author in G.nodes:
    if "total_citations" not in G.nodes[author]:
        G.nodes[author]["total_citations"] = 0
    if "first_publication_year" not in G.nodes[author]:
        G.nodes[author]["first_publication_year"] = None

# -------------------------
# 6. Save Network as JSON
# -------------------------
# Generate network data in node-link format
network_data = json_graph.node_link_data(G)  # 移除 edges="links"

# Save the network data as a JSON file
with open("Computational_Social_Scientists_Network.json", "w", encoding="utf-8") as f:
    json.dump(network_data, f, ensure_ascii=False, indent=4)


print("Network saved to Computational_Social_Scientists_Network.json")
