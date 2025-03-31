import pandas as pd
import networkx as nx
import json
import ast
from itertools import combinations
from netwulf import visualize
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

# -------------------------
# 1. 数据加载与列名修复（关键修复点）
# -------------------------
# 加载作者数据
authors_df = pd.read_csv("IC2S2_2024_Computational_Social_Scientists.csv")

# 修复列名（确保去除冒号后包含 h_index）
authors_df.columns = [col.rstrip(":") for col in authors_df.columns]
print("处理后的列名:", authors_df.columns.tolist())  # 确认包含 h_index

# 处理缺失值（防御式填充）
authors_df["h_index"] = authors_df["h_index"].fillna(0).astype(int)
authors_df["works_count"] = authors_df["works_count"].fillna(0).astype(int)
authors_df["country"] = authors_df.get("country--code", pd.Series("Unknown")).fillna("Unknown")

# 加载论文数据
papers_df = pd.read_csv("IC2S2_papers.csv")
papers_df["author_ids_parsed"] = papers_df["author_ids"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# -------------------------
# 2. 网络构建（强制属性初始化）
# -------------------------
G = nx.Graph()

for _, row in authors_df.iterrows():
    # 防御式属性获取（关键修复点）
    node_attrs = {
        "id": str(row["id"]),  # 确保 ID 为字符串
        "display_name": str(row.get("display_name", "Unknown")),
        "h_index": int(row.get("h_index", 0)),  # 强制存在该属性
        "works_count": int(row.get("works_count", 0)),
        "country": str(row.get("country", "Unknown"))
    }
    G.add_node(node_attrs["id"], **node_attrs)

# 构建合作边
edge_counter = {}
for _, row in papers_df.iterrows():
    authors = row["author_ids_parsed"]
    for u, v in combinations(authors, 2):
        edge = tuple(sorted((u, v)))
        edge_counter[edge] = edge_counter.get(edge, 0) + 1

G.add_weighted_edges_from([(u, v, w) for (u, v), w in edge_counter.items() if w > 0])

# -------------------------
# 3. 数据深度清洗（确保属性存在性）
# -------------------------
def enforce_attributes(G):
    """确保所有节点包含必需属性"""
    required_attrs = ["h_index", "works_count", "country", "display_name"]
    for node, data in G.nodes(data=True):
        for attr in required_attrs:
            if attr not in data:
                # 根据属性类型设置默认值
                default = 0 if attr in ["h_index", "works_count"] else "Unknown"
                data[attr] = default
        # 强制类型转换
        data["h_index"] = int(data["h_index"])
        data["works_count"] = int(data["works_count"])
        data["country"] = str(data["country"])
    return G

G = enforce_attributes(G)

# -------------------------
# 4. 提取最大连通组件
# -------------------------
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print(f"网络规模: 节点数 {G_lcc.number_of_nodes()}，边数 {G_lcc.number_of_edges()}")

# -------------------------
# 5. 可视化配置（数学精确的面积映射）
# -------------------------
# 计算 h_index 范围
h_values = [data["h_index"] for _, data in G_lcc.nodes(data=True)]
min_h, max_h = min(h_values), max(h_values)

config = {
    "NodeStyle": {
        "size": {
            "map": "h_index",
            "scale": (5, 25),
            "exponent": 0.5  # 面积正比于 h_index
        },
        "color": {
            "map": "country",
            "palette": "Category20"
        },
        "label": {
            "show": True,
            "fontSize": 6,
            "strokeColor": (1, 1, 1, 0.7)
        }
    },
    "EdgeStyle": {
        "color": (0.7, 0.7, 0.7, 0.2),
        "width": 0.3
    },
    "Layout": {
        "iterations": 1200,
        "gravity": 0.03,
        "charge": 15
    }
}

# 设置所有节点标签
for node in G_lcc.nodes:
    G_lcc.nodes[node]["label"] = G_lcc.nodes[node]["display_name"]

# -------------------------
# 6. 可视化执行
# -------------------------
try:
    # 新版参数
    visualize(G_lcc, config=config)  # 不传递保存路径
except TypeError:
    # 旧版兼容
    visualize(G_lcc, config=config)

# -------------------------
# 7. 保存网络数据
# -------------------------
with open("Computational_Social_Scientists_Network.json", "w") as f:
    json.dump(json_graph.node_link_data(G_lcc), f, indent=2)

# -------------------------
# 8. 使用 matplotlib 保存网络图像
# -------------------------
# 通过 matplotlib 保存图像
plt.figure(figsize=(12, 12))
plt.savefig("Computational_Social_Scientists_Network_Netwulf.png", format="PNG", dpi=300)
plt.close()
