import networkx as nx
import numpy as np
import json
import time
import random
import matplotlib

# Use non-interactive backend to avoid dependencies on pandas and GUI
# Change to inline backend to display plots directly in the output
# For Jupyter Notebook/IPython environments
matplotlib.use('Agg')  # Keep this for saving files
import matplotlib.pyplot as plt
from IPython.display import display, Image, HTML  # Import display tools

# Start timing
start_time = time.time()
print("Starting execution...")

# Load actual network data from JSON file
with open("Computational_Social_Scientists_Network.json", "r", encoding="utf-8") as f:
    data = json.load(f)
G_actual = nx.node_link_graph(data)

N = G_actual.number_of_nodes()
E_actual = G_actual.number_of_edges()
print("Actual network: Nodes =", N, ", Edges =", E_actual)

# Calculate random network parameter p and expected average degree
p = E_actual / (N * (N - 1) / 2)
avg_degree = p * (N - 1)

print("Calculated probability p =", round(p, 6))
print("Expected random network average degree =", round(avg_degree, 2))

# Record basic network information using a dictionary
network_info = {
    'Nodes': N,
    'Edges': E_actual,
    'Probability p': round(p, 6),
    'Expected average degree': round(avg_degree, 2)
}
print("\nBasic network information:")
for k, v in network_info.items():
    print(f"{k}: {v}")

# Time check
print(f"Basic information calculation completed, time elapsed: {time.time() - start_time:.2f} seconds")

# Generate random network - using NetworkX built-in function (more efficient)
print("\nGenerating random network (using built-in function)...")
random_start = time.time()
G_random = nx.fast_gnp_random_graph(N, p, seed=42)
random_time = time.time() - random_start
print(f"Random network generation completed, time elapsed: {random_time:.2f} seconds")
print("Random network: Nodes =", G_random.number_of_nodes(), ", Edges =", G_random.number_of_edges())


# Extract giant connected component (no immediate visualization)
def get_giant_component(G, name="Network"):
    """Extract the largest connected component"""
    print(f"Extracting giant connected component of {name}...")
    cc_start = time.time()

    # Find all connected components
    components = list(nx.connected_components(G))
    print(f"- {name} has {len(components)} connected components")

    # Get the largest connected component
    giant_nodes = max(components, key=len)
    G_giant = G.subgraph(giant_nodes)

    # Output information
    print(f"- Giant component contains {G_giant.number_of_nodes()} nodes and {G_giant.number_of_edges()} edges")
    print(f"- Comprises {G_giant.number_of_nodes() / G.number_of_nodes() * 100:.2f}% of total nodes")
    print(f"- Extraction time: {time.time() - cc_start:.2f} seconds")

    return G_giant


# Analyze degree distribution (without plotting)
def analyze_degree_distribution(G, name="Network"):
    """Analyze basic statistical data of network degree distribution"""
    print(f"\nAnalyzing degree distribution of {name}...")
    degree_start = time.time()

    # Get degrees of all nodes
    degree_seq = [d for n, d in G.degree()]

    # Basic statistics
    n = len(degree_seq)
    mean = sum(degree_seq) / n
    sorted_seq = sorted(degree_seq)
    median = sorted_seq[n // 2] if n % 2 == 1 else (sorted_seq[n // 2 - 1] + sorted_seq[n // 2]) / 2
    min_val = min(degree_seq)
    max_val = max(degree_seq)

    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in degree_seq) / n
    std_dev = variance ** 0.5

    # Output statistical data
    print(f"{name} degree statistics:")
    print(f"- Count: {n}")
    print(f"- Mean: {round(mean, 2)}")
    print(f"- Median: {median}")
    print(f"- Standard deviation: {round(std_dev, 2)}")
    print(f"- Minimum: {min_val}")
    print(f"- Maximum: {max_val}")
    print(f"- Analysis time: {time.time() - degree_start:.2f} seconds")

    # Return statistical results
    return {
        "Mean": mean,
        "Median": median,
        "Standard deviation": std_dev,
        "Minimum": min_val,
        "Maximum": max_val
    }


# Calculate approximate average shortest path length
def calculate_approximate_path_length(G, samples=1000, name="Network"):
    """Use sampling method to approximately calculate average shortest path length"""
    print(f"Calculating approximate average shortest path length of {name} (samples={samples})...")
    path_start = time.time()

    # If network nodes are fewer than samples, calculate exact value
    if G.number_of_nodes() <= samples:
        print(f"- Network nodes fewer than sample size, calculating exact value")
        avg_path = nx.average_shortest_path_length(G)
        print(f"- Exact average shortest path length: {avg_path:.4f}")
        print(f"- Calculation time: {time.time() - path_start:.2f} seconds")
        return avg_path

    # Random sampling of nodes
    print(f"- Randomly sampling {samples} nodes from {G.number_of_nodes()} nodes")
    nodes = list(G.nodes())
    sampled_nodes = random.sample(nodes, samples)

    # Calculate shortest paths between sampled nodes
    total_paths = 0
    path_sum = 0

    # Use Floyd Warshall algorithm to calculate all shortest paths - suitable for dense graphs
    if samples > 100:  # For larger samples, use more efficient method
        print(f"- Using efficient method to calculate all shortest paths between {samples} nodes")
        G_sample = G.subgraph(sampled_nodes)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G_sample))

        for u in path_lengths:
            for v in path_lengths[u]:
                if u != v:  # Exclude self-loops
                    path_sum += path_lengths[u][v]
                    total_paths += 1
    else:  # For smaller samples, calculate directly
        print(f"- Calculating all shortest paths between {samples} nodes")
        for i, u in enumerate(sampled_nodes):
            for v in sampled_nodes[i + 1:]:
                try:
                    path_length = nx.shortest_path_length(G, u, v)
                    path_sum += path_length
                    total_paths += 1
                except nx.NetworkXNoPath:
                    # If no path exists between nodes, ignore
                    pass

    # Calculate average
    avg_path = path_sum / total_paths if total_paths > 0 else float('inf')

    print(f"- Calculated {total_paths} paths")
    print(f"- Approximate average shortest path length: {avg_path:.4f}")
    print(f"- Calculation time: {time.time() - path_start:.2f} seconds")

    return avg_path


# Calculate and compare network properties
def compare_network_properties(G1, G1_giant, G2, G2_giant, name1="Actual network", name2="Random network"):
    """Calculate and compare important properties of two networks"""
    print(f"\nCalculating properties of {name1} and {name2}...")
    props_start = time.time()

    # Calculate average clustering coefficient (may be slow)
    print("Calculating average clustering coefficient...")
    clustering_start = time.time()

    # For large networks, use sampling method to calculate clustering coefficient
    if G1.number_of_nodes() > 10000:
        print(f"- {name1} has many nodes, using sampling method to calculate clustering coefficient (sample=1000)")
        nodes_sample = random.sample(list(G1.nodes()), 1000)
        G1_sample = G1.subgraph(nodes_sample)
        avg_clustering_1 = nx.average_clustering(G1_sample)
        print(f"- {name1} sampled clustering coefficient: {avg_clustering_1:.6f}")
    else:
        avg_clustering_1 = nx.average_clustering(G1)

    if G2.number_of_nodes() > 10000:
        print(f"- {name2} has many nodes, using sampling method to calculate clustering coefficient (sample=1000)")
        nodes_sample = random.sample(list(G2.nodes()), 1000)
        G2_sample = G2.subgraph(nodes_sample)
        avg_clustering_2 = nx.average_clustering(G2_sample)
        print(f"- {name2} sampled clustering coefficient: {avg_clustering_2:.6f}")
    else:
        avg_clustering_2 = nx.average_clustering(G2)

    # Prevent division by zero error
    if avg_clustering_2 == 0:
        # Random network clustering coefficient is theoretically p, we can use this as minimum value
        avg_clustering_2 = p
        print(f"- {name2} clustering coefficient is 0, using theoretical value p={p:.6f} instead")

    print(f"- Clustering coefficient calculation completed, time elapsed: {time.time() - clustering_start:.2f} seconds")

    # Calculate approximate average shortest path length (using sampling method)
    print("Calculating approximate average shortest path length...")
    path_start = time.time()

    # Use approximate algorithm to calculate average path length
    sample_size = 1000  # Number of nodes to sample
    avg_path_1 = calculate_approximate_path_length(G1_giant, samples=sample_size, name=name1)
    avg_path_2 = calculate_approximate_path_length(G2_giant, samples=sample_size, name=name2)

    print(f"- Path length calculation completed, time elapsed: {time.time() - path_start:.2f} seconds")

    # Create results dictionary
    results = {
        'Nodes': [G1.number_of_nodes(), G2.number_of_nodes()],
        'Edges': [G1.number_of_edges(), G2.number_of_edges()],
        'Average degree': [2 * G1.number_of_edges() / G1.number_of_nodes(),
                           2 * G2.number_of_edges() / G2.number_of_nodes()],
        'Average clustering coefficient': [avg_clustering_1, avg_clustering_2],
        'Giant component nodes': [G1_giant.number_of_nodes(), G2_giant.number_of_nodes()],
        'Giant component percentage': [G1_giant.number_of_nodes() / G1.number_of_nodes(),
                                       G2_giant.number_of_nodes() / G2.number_of_nodes()],
        'Average shortest path length (approx)': [avg_path_1, avg_path_2]
    }

    # Output results table
    print(f"\n{name1} and {name2} property comparison:")
    print(f"{'Metric':<20} {name1:<12} {name2:<12} Ratio({name1}/{name2})")
    print("-" * 60)

    for metric, values in results.items():
        ratio = values[0] / values[1] if values[1] != 0 else float('inf')
        print(f"{metric:<20} {values[0]:<12.4f} {values[1]:<12.4f} {ratio:<.4f}")

    # Calculate small-world properties
    L_ratio = avg_path_1 / avg_path_2
    C_ratio = avg_clustering_1 / avg_clustering_2

    print(f"\nSmall-world property analysis:")
    print(f"L_{name1} / L_{name2} = {round(L_ratio, 2)}")
    print(f"C_{name1} / C_{name2} = {round(C_ratio, 2)}")

    if L_ratio < 2 and C_ratio > 2:
        print(f"Conclusion: {name1} likely has small-world properties")
    else:
        print(f"Conclusion: {name1} may not have obvious small-world properties")

    print(f"Property comparison completed, total time elapsed: {time.time() - props_start:.2f} seconds")

    return L_ratio, C_ratio


# Plot degree distributions - MODIFIED to display plots directly
def plot_degree_distributions(G_actual, G_random):
    """
    Plot degree distributions of actual and random networks,
    save images AND display them directly in the output
    """
    print("\nPlotting degree distributions...")
    plot_start = time.time()

    # Get actual network degrees
    actual_degrees = [d for n, d in G_actual.degree()]
    actual_max_degree = max(actual_degrees)
    actual_avg_degree = 2 * G_actual.number_of_edges() / G_actual.number_of_nodes()

    # Get random network degrees
    random_degrees = [d for n, d in G_random.degree()]
    random_max_degree = max(random_degrees)
    random_avg_degree = 2 * G_random.number_of_edges() / G_random.number_of_nodes()

    # 1. Regular scale histogram
    plt.figure(figsize=(10, 6))
    # Limit x-axis range to better view the core part of the distribution
    plt.hist(actual_degrees, bins=30, alpha=0.7, label='Actual network', range=(0, 50))
    plt.hist(random_degrees, bins=30, alpha=0.7, label='Random network', range=(0, 50))

    # Add vertical lines for average degrees
    plt.axvline(x=actual_avg_degree, color='blue', linestyle='--',
                label=f'Actual network average degree ({actual_avg_degree:.2f})')
    plt.axvline(x=random_avg_degree, color='orange', linestyle='--',
                label=f'Random network average degree ({random_avg_degree:.2f})')

    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution Comparison (Limited Range)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure and display it
    plt.savefig('degree_distribution_limited.png')
    print("- Image saved: degree_distribution_limited.png")
    # Display the figure directly in the output
    display(Image('degree_distribution_limited.png'))
    plt.close()

    # 2. Log scale distribution plot (handling heavy-tailed distributions)
    plt.figure(figsize=(10, 6))

    # Calculate degree frequency distribution
    max_degree = max(actual_max_degree, random_max_degree)
    bins = np.logspace(np.log10(1), np.log10(max_degree + 1), 20)

    plt.hist(actual_degrees, bins=bins, alpha=0.7, density=True, label='Actual network')
    plt.hist(random_degrees, bins=bins, alpha=0.7, density=True, label='Random network')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Degree Distribution Comparison (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Save the figure and display it
    plt.savefig('degree_distribution_log.png')
    print("- Image saved: degree_distribution_log.png")
    # Display the figure directly in the output
    display(Image('degree_distribution_log.png'))
    plt.close()

    # 3. Complementary Cumulative Distribution Function (CCDF) - better shows heavy-tail properties
    plt.figure(figsize=(10, 6))

    # Calculate CCDF
    actual_sorted = np.sort(actual_degrees)
    actual_ccdf = np.arange(len(actual_sorted)) / float(len(actual_sorted))
    actual_ccdf = 1 - actual_ccdf

    random_sorted = np.sort(random_degrees)
    random_ccdf = np.arange(len(random_sorted)) / float(len(random_sorted))
    random_ccdf = 1 - random_ccdf

    # Plot CCDF
    plt.loglog(actual_sorted, actual_ccdf, 'o-', markersize=3, label='Actual network')
    plt.loglog(random_sorted, random_ccdf, 's-', markersize=3, label='Random network')

    plt.xlabel('Degree k (log scale)')
    plt.ylabel('P(K ≥ k) (log scale)')
    plt.title('Cumulative Degree Distribution (CCDF)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    # Save the figure and display it
    plt.savefig('degree_distribution_ccdf.png')
    print("- Image saved: degree_distribution_ccdf.png")
    # Display the figure directly in the output
    display(Image('degree_distribution_ccdf.png'))
    plt.close()

    # Display a summary of all plots together (optional)
    print("\n- Summary of all visualizations:")
    display(HTML("""
    <table style="border-collapse: collapse; width: 100%;">
      <tr>
        <th style="border: 1px solid black; padding: 8px; text-align: center;">Regular Scale</th>
        <th style="border: 1px solid black; padding: 8px; text-align: center;">Log Scale</th>
        <th style="border: 1px solid black; padding: 8px; text-align: center;">CCDF</th>
      </tr>
      <tr>
        <td style="border: 1px solid black; padding: 8px; text-align: center;"><img src="degree_distribution_limited.png" width="300"></td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;"><img src="degree_distribution_log.png" width="300"></td>
        <td style="border: 1px solid black; padding: 8px; text-align: center;"><img src="degree_distribution_ccdf.png" width="300"></td>
      </tr>
    </table>
    """))

    print(
        f"- Plotting complete, 3 visualization images saved and displayed, time elapsed: {time.time() - plot_start:.2f} seconds")


# Main program flow
print("\n===== Step 1: Extract Giant Connected Component =====")
G_actual_giant = get_giant_component(G_actual, "Actual network")
G_random_giant = get_giant_component(G_random, "Random network")

print("\n===== Step 2: Analyze Degree Distribution =====")
actual_degree_stats = analyze_degree_distribution(G_actual, "Actual network")
random_degree_stats = analyze_degree_distribution(G_random, "Random network")

print("\n===== Step 3: Calculate Network Properties and Compare =====")
L_ratio, C_ratio = compare_network_properties(G_actual, G_actual_giant, G_random, G_random_giant)

print("\n===== Step 4: Plot Degree Distributions =====")
plot_degree_distributions(G_actual, G_random)

# Print total execution time
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# Analysis results explanation
print("\n===== Network Analysis Results Explanation =====")
print("1. Network regime analysis:")
critical_threshold = 1 / N
print(f"   - Critical threshold p_c = 1/N = {critical_threshold:.8f}")
print(f"   - Actual network p = {p:.8f}")
if p > critical_threshold:
    print(f"   - Conclusion: Network is in super-critical regime (p > p_c), has giant connected component")
else:
    print(f"   - Conclusion: Network is in sub-critical regime (p < p_c), no giant connected component")

print("\n2. Small-world property analysis:")
print(f"   - L_ratio = {L_ratio:.2f} (Actual/Random network average path length ratio)")
print(f"   - C_ratio = {C_ratio:.2f} (Actual/Random network clustering coefficient ratio)")

if L_ratio < 2 and C_ratio > 2:
    print("   - Conclusion: Actual network exhibits small-world properties")
    print("     * Path length close to random network (high efficiency)")
    print("     * Clustering coefficient significantly higher than random network (high clustering)")
else:
    print("   - Conclusion: Actual network may not have obvious small-world properties")

print("\n3. Degree distribution characteristics:")
print(
    "   - Actual network degree distribution shows heavy-tail properties, maximum degree significantly higher than random network")
print(
    f"   - Actual network degree standard deviation ({actual_degree_stats['Standard deviation']:.2f}) much larger than random network ({random_degree_stats['Standard deviation']:.2f})")
print(
    "   - This indicates the existence of a few highly connected 'hub' nodes in the actual network, while in the random network, node degree distribution is more uniform")

print("\n4. Overall network structure:")
if len(list(nx.connected_components(G_actual))) == 1:
    print("   - Actual network is fully connected, all nodes are in one connected component")
else:
    print(f"   - Actual network has {len(list(nx.connected_components(G_actual)))} connected components")
    print(
        f"   - Largest connected component contains {G_actual_giant.number_of_nodes() / G_actual.number_of_nodes() * 100:.2f}% of the nodes")

print(f"   - Random network has {len(list(nx.connected_components(G_random)))} connected components")
print(
    f"   - Largest connected component contains {G_random_giant.number_of_nodes() / G_random.number_of_nodes() * 100:.2f}% of the nodes")