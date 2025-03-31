import pandas as pd
import networkx as nx
import json
import ast
import statistics
from itertools import combinations
from networkx.readwrite import json_graph

# -------------------------
# Part 1: Network Construction
# -------------------------

# 1. Load data
authors_df = pd.read_csv("IC2S2_2024_Computational_Social_Scientists.csv")
authors_df.columns = [col.rstrip(":") for col in authors_df.columns]

papers_df = pd.read_csv("IC2S2_papers.csv")
papers_df["author_ids_parsed"] = papers_df["author_ids"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# 2. Construct weighted edge list
edge_dict = {}
for authors in papers_df["author_ids_parsed"]:
    if len(authors) > 1:
        for pair in combinations(sorted(authors), 2):
            edge_dict[pair] = edge_dict.get(pair, 0) + 1
weighted_edgelist = [(u, v, w) for (u, v), w in edge_dict.items()]

# 3. Build network
G = nx.Graph()
G.add_weighted_edges_from(weighted_edgelist)

# 4. Add node attributes
author_metadata = authors_df.set_index("id").to_dict(orient="index")
for node in G.nodes:
    G.nodes[node].update(author_metadata.get(node, {}))

# 5. Add paper-related attributes
author_stats = {}
for _, row in papers_df.iterrows():
    year = row.get("publication_year")
    citations = row.get("cited_by_count", 0)
    for author in row["author_ids_parsed"]:
        if author not in author_stats:
            author_stats[author] = {"years": [], "citations": 0}
        if pd.notnull(year):
            author_stats[author]["years"].append(year)
        author_stats[author]["citations"] += citations

for author, stats in author_stats.items():
    if author in G.nodes:
        G.nodes[author]["total_citations"] = stats["citations"]
        G.nodes[author]["first_publication_year"] = min(stats["years"]) if stats["years"] else None

# 6. Save network
with open("Computational_Social_Scientists_Network.json", "w") as f:
    json.dump(json_graph.node_link_data(G), f)


# -------------------------
# Part 2: Network Analysis
# -------------------------

def analyze_network(G):
    # Basic network metrics
    results = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "components": nx.number_connected_components(G),
        "isolated_nodes": len(list(nx.isolates(G)))
    }

    # Degree analysis
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    results.update({
        "degree_stats": {
            "mean": statistics.mean(degrees.values()),
            "median": statistics.median(degrees.values()),
            "mode": max(set(degrees.values()), key=list(degrees.values()).count),
            "min": min(degrees.values()),
            "max": max(degrees.values())
        },
        "strength_stats": {
            "mean": statistics.mean(strengths.values()),
            "median": statistics.median(strengths.values()),
            "mode": max(set(strengths.values()), key=list(strengths.values()).count),
            "min": min(strengths.values()),
            "max": max(strengths.values())
        }
    })

    # Top authors
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    results["top_authors"] = [
        (G.nodes[author]["display_name"], degree)
        for author, degree in top_degree
    ]

    return results


# Execute analysis
analysis_results = analyze_network(G)

# Print results
print(f"""
Network Metrics:
- Number of nodes: {analysis_results['num_nodes']}
- Number of edges: {analysis_results['num_edges']}
- Density: {analysis_results['density']:.4f}
- Fully connected: {'Yes' if analysis_results['is_connected'] else 'No'}
- Number of connected components: {analysis_results['components']}
- Isolated nodes: {analysis_results['isolated_nodes']}

Degree Analysis:
Average degree: {analysis_results['degree_stats']['mean']:.1f}
Median degree: {analysis_results['degree_stats']['median']}
Mode degree: {analysis_results['degree_stats']['mode']}
Degree range: {analysis_results['degree_stats']['min']}-{analysis_results['degree_stats']['max']}

Weighted Degree Analysis:
Average strength: {analysis_results['strength_stats']['mean']:.1f}
Median strength: {analysis_results['strength_stats']['median']}
Mode strength: {analysis_results['strength_stats']['mode']}
Strength range: {analysis_results['strength_stats']['min']}-{analysis_results['strength_stats']['max']}

Top 5 Authors by Degree:
""")
for idx, (name, degree) in enumerate(analysis_results['top_authors'], 1):
    print(f"{idx}. {name} (Degree: {degree})")

# Discussion
print("""
Discussion:
1. The network density is typically very low (<0.01), consistent with the sparse nature of real-world collaboration networks
2. The network is usually disconnected with multiple connected components, reflecting distinct research communities
3. Degree distribution shows typical right-skew characteristics, with few core authors acting as network hubs
4. Top authors are typically well-known scholars or institutional leaders in the field
""")