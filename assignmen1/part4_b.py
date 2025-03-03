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

# 1. 加载数据
authors_df = pd.read_csv("IC2S2_2024_Computational_Social_Scientists.csv")
authors_df.columns = [col.rstrip(":") for col in authors_df.columns]

papers_df = pd.read_csv("IC2S2_papers.csv")
papers_df["author_ids_parsed"] = papers_df["author_ids"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# 2. 构建加权边列表
edge_dict = {}
for authors in papers_df["author_ids_parsed"]:
    if len(authors) > 1:
        for pair in combinations(sorted(authors), 2):
            edge_dict[pair] = edge_dict.get(pair, 0) + 1
weighted_edgelist = [(u, v, w) for (u, v), w in edge_dict.items()]

# 3. 构建网络
G = nx.Graph()
G.add_weighted_edges_from(weighted_edgelist)

# 4. 添加节点属性
author_metadata = authors_df.set_index("id").to_dict(orient="index")
for node in G.nodes:
    G.nodes[node].update(author_metadata.get(node, {}))

# 5. 添加论文相关属性
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

# 6. 保存网络
with open("Computational_Social_Scientists_Network.json", "w") as f:
    json.dump(json_graph.node_link_data(G), f)


# -------------------------
# Part 2: Network Analysis
# -------------------------

def analyze_network(G):
    # 网络基本指标
    results = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "components": nx.number_connected_components(G),
        "isolated_nodes": len(list(nx.isolates(G)))  # 修正括号
    }

    # 度分析
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

    # 顶尖作者
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    results["top_authors"] = [
        (G.nodes[author]["display_name"], degree)
        for author, degree in top_degree
    ]

    return results


# 执行分析
analysis_results = analyze_network(G)

# 打印结果
print(f"""
Network Metrics:
- 节点数: {analysis_results['num_nodes']}
- 边数: {analysis_results['num_edges']}
- 密度: {analysis_results['density']:.4f}
- 全连接: {'是' if analysis_results['is_connected'] else '否'}
- 连通组件数: {analysis_results['components']}
- 孤立节点数: {analysis_results['isolated_nodes']}

Degree Analysis:
平均度: {analysis_results['degree_stats']['mean']:.1f}
度中位数: {analysis_results['degree_stats']['median']}
度众数: {analysis_results['degree_stats']['mode']}
度范围: {analysis_results['degree_stats']['min']}-{analysis_results['degree_stats']['max']}

加权度分析:
平均强度: {analysis_results['strength_stats']['mean']:.1f}
强度中位数: {analysis_results['strength_stats']['median']}
强度众数: {analysis_results['strength_stats']['mode']}
强度范围: {analysis_results['strength_stats']['min']}-{analysis_results['strength_stats']['max']}

Top 5 Authors by Degree:
""")
for idx, (name, degree) in enumerate(analysis_results['top_authors'], 1):
    print(f"{idx}. {name} (Degree: {degree})")

# 结果讨论
print("""
讨论:
1. 网络密度通常非常低（<0.01），符合真实世界合作网络的稀疏特性
2. 网络通常不连通，存在多个连通组件，反映不同的研究团体
3. 度分布呈现典型的右偏特征，少数核心作者作为网络枢纽存在
4. 顶尖作者通常是领域内的知名学者或机构领导者
""")
#Part 1: Network Metrics
#
# 1. What is the total number of nodes (authors) and links (collaborations) in the network?
#
# Answer:
# Nodes: 14,920 authors
# Links: 55,466 collaborations
# 2. Calculate the network's density. Would you say that the network is sparse?
#
# Answer:
# Density: 0.0005
# The network is extremely sparse (density << 0.01). This is expected for large academic networks, as researchers typically collaborate with a tiny fraction of the total community.
# 3. Is the network fully connected or disconnected?
#
# Answer:
# Disconnected (全连接: 否).
# 4. If disconnected, how many connected components does it have?
#
# Answer:
# 119 connected components, reflecting distinct research clusters (e.g., institutional groups or thematic subfields).
# 5. How many isolated nodes are there?
#
# Answer:
# 0 isolated nodes, suggesting the dataset excludes authors with no collaborations.
# 6. Discuss network density and connectivity (150 words):
#
# The observed metrics align with expectations:
#
# Ultra-low density (0.0005) confirms collaboration sparsity in large academic communities.
# 119 components indicate fragmented collaboration patterns, likely driven by institutional/organizational boundaries.
# No isolates suggests data curation (e.g., filtering authors with ≥1 paper).
# The structure mirrors real-world scientific networks where researchers primarily collaborate within local groups. This fragmentation may limit knowledge diffusion across components.
# Part 2: Degree Analysis
#
# Compute degree and strength metrics. What do they reveal?
#
# Metric	Degree	Weighted Degree
# Average	7.4	11.6
# Median	6.0	6.0
# Mode	5	8
# Range	1–1,147	1–2,708
# Interpretation (150 words):
#
# Right-skewed distribution (mean > median, max ≫ mean) indicates a scale-free network, dominated by hyper-connected hubs.
# Median degree=6: 50% of authors have ≤6 collaborators.
# Max degree=1,147 reveals extreme hubs (e.g., Robert West) acting as network glue.
# Weighted degree metrics (average=11.6 vs degree=7.4) show repeated collaborations ("strong ties").
# Max strength=2,708 suggests a small cohort of authors engage in institutionalized team science.
# These patterns reflect "preferential attachment" in academic collaboration, where established authors attract new partners.
# Part 3: Top Authors
#
# 1. Top 5 authors by degree:
#
# Robert West (Degree=1,147)
# Yasuhiro Suzuki (501)
# Xiao Zhang (431)
# Ke Li (347)
# Jingwen Zhang (309)
# 2. Their network role:
#
# These superhubs act as:
#
# Bridges between research communities
# Anchors for large consortia (e.g., multi-institutional projects)
# Knowledge brokers facilitating interdisciplinary exchange
# 3. Research alignment with CSS (150 words):
#
# Robert West (EPFL) specializes in computational social science, aligning with CSS themes.
# Yasuhiro Suzuki (Nagoya) focuses on evolutionary computation, which may connect to CSS through methodology development.
# Xiao Zhang (Beijing) works on social network analysis, directly relevant to CSS.
# Ke Li and Jingwen Zhang (Chinese Academy of Sciences) publish on machine learning applications, which may diverge from CSS if focused purely on technical methods.
# Potential discrepancies arise from:
# Interdisciplinary collaborations with computer scientists
# Database inclusion of technical contributors (e.g., data engineers)
# Methodological focus without direct social science applications
# ##