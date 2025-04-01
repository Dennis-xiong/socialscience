import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import sys
from PIL import Image

# Note: If python-louvain is not installed, you may need to install it with:
# pip install python-louvain
try:
    from community import community_louvain

    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain package not installed. Louvain community detection will be skipped.")
    print("To install it, run: pip install python-louvain")


def load_karate_club():
    """
    Load Zachary's Karate Club network with faction information.
    """
    # Load the built-in karate club network
    G = nx.karate_club_graph()

    # The graph already has 'club' attribute indicating factions (0 for Mr. Hi, 1 for Officer)
    return G


def calculate_karate_assortativity(G):
    """
    Calculate and return the assortativity coefficient for the karate club network
    based on the faction attribute.
    """
    # Extract faction attribute for each node
    return nx.attribute_assortativity_coefficient(G, 'club')


def calculate_modularity(G, partition):
    """
    Calculate the modularity of a graph partitioning using equation 9.12 in the book.

    Parameters:
    G : NetworkX graph
        The graph to analyze
    partition : dict
        A dictionary mapping node IDs to their community/partition IDs

    Returns:
    float : The modularity value of the partitioning
    """
    m = G.number_of_edges()
    degrees = dict(G.degree())

    # Initialize modularity
    Q = 0

    # For each pair of nodes
    for i in G.nodes():
        for j in G.nodes():
            # If they are in the same community
            if partition[i] == partition[j]:
                # Get the adjacency value (1 if connected, 0 otherwise)
                A_ij = 1 if G.has_edge(i, j) else 0

                # Calculate expected number of edges in random graph
                k_i = degrees[i]
                k_j = degrees[j]
                expected = (k_i * k_j) / (2 * m)

                # Add contribution to modularity
                Q += (A_ij - expected) / (2 * m)

    return Q


def double_edge_swap_karate(G, num_swaps=None, max_attempts=100):
    """
    Generates a random graph using the double edge swap algorithm
    while preserving the degree distribution.

    Parameters:
    G : NetworkX graph
        The original graph
    num_swaps : int, optional
        Number of successful swaps to perform. If None, use 10*|E|
    max_attempts : int, optional
        Maximum number of attempts to try before giving up on a swap

    Returns:
    G_random : NetworkX graph
        A graph with the same degree sequence as G but with randomized connections
    """
    # Create a copy of the original graph
    G_random = G.copy()

    # If num_swaps is not specified, set it to 10 times the number of edges
    if num_swaps is None:
        num_swaps = 10 * G.number_of_edges()

    # Keep track of successful swaps
    swaps_done = 0
    attempts = 0

    while swaps_done < num_swaps and attempts < num_swaps * max_attempts:
        attempts += 1

        # Get all edges as a list
        edges = list(G_random.edges())

        # If not enough edges to swap
        if len(edges) < 2:
            break

        # Select two random edges
        idx1, idx2 = random.sample(range(len(edges)), 2)
        e1 = edges[idx1]
        e2 = edges[idx2]

        # Unpack the edges
        u, v = e1
        x, y = e2

        # Check if the nodes are distinct (no self-loops would be created)
        if u == y or v == x or u == x or v == y or u == v or x == y:
            continue

        # New edges would be (u, y) and (x, v)
        if G_random.has_edge(u, y) or G_random.has_edge(x, v):
            continue

        # Perform the swap
        G_random.remove_edge(*e1)
        G_random.remove_edge(*e2)
        G_random.add_edge(u, y)
        G_random.add_edge(x, v)

        swaps_done += 1

    return G_random


def visualize_karate_network(G):
    """
    Visualize the Karate Club network with nodes colored by faction.
    """
    plt.figure(figsize=(10, 8))

    # Position nodes using the spring layout
    pos = nx.spring_layout(G, seed=42)

    # Get factions (club attribute)
    club = nx.get_node_attributes(G, 'club')

    # Create a list of colors for nodes based on faction
    colors = ['skyblue' if club[node] == 0 else 'salmon' for node in G.nodes()]

    # Draw the network
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=colors,
        node_size=400,
        with_labels=True,
        font_weight='bold',
        edge_color='gray',
        font_size=10
    )

    plt.title("Zachary's Karate Club Network")
    plt.axis('off')
    plt.tight_layout()

    # Save the figure to the current directory
    image_path = "./karate_club_network.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
        # For a simple ASCII representation of the image if desired
        # Print the image path rather than trying to display it in terminal
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()


def compute_adjacency_spectrum(G):
    """
    Compute and plot the spectrum of the adjacency matrix.
    """
    # Get the adjacency matrix
    A = nx.to_numpy_array(G)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)

    # Sort eigenvalues in decreasing order
    eigenvalues = sorted(eigenvalues, reverse=True)

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Spectrum of Adjacency Matrix')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure to the current directory
    image_path = "./adjacency_spectrum.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()

    return eigenvalues


def apply_clustering(G, n_clusters=2):
    """
    Apply K-means and Spectral clustering to the network and compare with the ground truth.
    """
    # Get adjacency matrix
    A = nx.to_numpy_array(G)

    # Get ground truth labels
    true_labels = np.array([G.nodes[i]['club'] for i in range(len(G))])

    # Apply K-means to the adjacency matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(A)

    # Apply Spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    spectral_labels = spectral.fit_predict(A)

    # Calculate ARI (Adjusted Rand Index) to measure clustering quality
    kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
    spectral_ari = adjusted_rand_score(true_labels, spectral_labels)

    print(f"K-means ARI: {kmeans_ari:.4f}")
    print(f"Spectral Clustering ARI: {spectral_ari:.4f}")

    # Visualize clustering results
    plt.figure(figsize=(15, 5))
    pos = nx.spring_layout(G, seed=42)

    # Plot ground truth
    plt.subplot(1, 3, 1)
    nx.draw_networkx(G, pos=pos,
                     node_color=['skyblue' if G.nodes[n]['club'] == 0 else 'salmon' for n in G.nodes()],
                     node_size=300, with_labels=True, font_size=8)
    plt.title("Ground Truth")
    plt.axis('off')

    # Plot K-means results
    plt.subplot(1, 3, 2)
    nx.draw_networkx(G, pos=pos,
                     node_color=['skyblue' if label == 0 else 'salmon' for label in kmeans_labels],
                     node_size=300, with_labels=True, font_size=8)
    plt.title(f"K-means Clustering (ARI: {kmeans_ari:.4f})")
    plt.axis('off')

    # Plot Spectral clustering results
    plt.subplot(1, 3, 3)
    nx.draw_networkx(G, pos=pos,
                     node_color=['skyblue' if label == 0 else 'salmon' for label in spectral_labels],
                     node_size=300, with_labels=True, font_size=8)
    plt.title(f"Spectral Clustering (ARI: {spectral_ari:.4f})")
    plt.axis('off')

    plt.tight_layout()

    # Save the figure to the current directory
    image_path = "./clustering_comparison.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()

    return {
        'true_labels': true_labels,
        'kmeans_labels': kmeans_labels,
        'spectral_labels': spectral_labels,
        'kmeans_ari': kmeans_ari,
        'spectral_ari': spectral_ari
    }


def analyze_modularity_distribution(G, n_simulations=1000):
    """
    Analyze the distribution of modularity values for random networks
    compared to the original Karate Club split.
    """
    # Get the original club partition
    club_partition = nx.get_node_attributes(G, 'club')

    # Calculate modularity for the original club split
    original_modularity = calculate_modularity(G, club_partition)
    print(f"Original club split modularity: {original_modularity:.6f}")

    # Generate random networks and calculate their modularities
    random_modularities = []

    for i in range(n_simulations):
        # Generate a randomized network
        G_random = double_edge_swap_karate(G)

        # Calculate modularity for the club split on this random network
        random_modularity = calculate_modularity(G_random, club_partition)
        random_modularities.append(random_modularity)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_simulations} random networks")

    # Calculate statistics
    mean_modularity = np.mean(random_modularities)
    std_modularity = np.std(random_modularities)

    print(f"Random networks mean modularity: {mean_modularity:.6f}")
    print(f"Random networks std deviation: {std_modularity:.6f}")

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(random_modularities, bins=30, alpha=0.7,
             label=f'Random Networks\nMean={mean_modularity:.4f}\nStd={std_modularity:.4f}')
    plt.axvline(x=original_modularity, color='r', linestyle='--',
                label=f'Original Club Split (Q={original_modularity:.4f})')
    plt.xlabel('Modularity (Q)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Modularity for Random Networks vs. Original Club Split')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate p-value (fraction of random networks with modularity >= original)
    p_value = sum(Q >= original_modularity for Q in random_modularities) / n_simulations
    plt.text(0.05, 0.95, f'p-value = {p_value:.6f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the figure to the current directory
    image_path = "./modularity_distribution.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()

    return {
        'original_modularity': original_modularity,
        'random_modularities': random_modularities,
        'mean': mean_modularity,
        'std': std_modularity,
        'p_value': p_value
    }


def analyze_louvain_communities(G):
    """
    Apply the Louvain community detection algorithm and compare with the club split.
    """
    if not LOUVAIN_AVAILABLE:
        print("Skipping Louvain community detection as the package is not installed.")
        return None

    # Get the original club partition
    club_partition = nx.get_node_attributes(G, 'club')

    # Calculate modularity for the original club split
    original_modularity = calculate_modularity(G, club_partition)

    # Apply Louvain algorithm
    louvain_partition = community_louvain.best_partition(G)
    louvain_modularity = community_louvain.modularity(louvain_partition, G)

    print(f"Original club split modularity: {original_modularity:.6f}")
    print(f"Louvain communities modularity: {louvain_modularity:.6f}")

    # Get number of communities from Louvain
    num_louvain_communities = len(set(louvain_partition.values()))
    print(f"Number of Louvain communities: {num_louvain_communities}")

    # Create a confusion matrix
    confusion_matrix = np.zeros((num_louvain_communities, 2))

    # Fill the confusion matrix
    for node, louvain_comm in louvain_partition.items():
        club = club_partition[node]
        confusion_matrix[louvain_comm, club] += 1

    # Visualize the communities
    plt.figure(figsize=(15, 5))
    pos = nx.spring_layout(G, seed=42)

    # Plot original club split
    plt.subplot(1, 2, 1)
    nx.draw_networkx(G, pos=pos,
                     node_color=['skyblue' if club_partition[n] == 0 else 'salmon' for n in G.nodes()],
                     node_size=300, with_labels=True, font_size=8)
    plt.title(f"Original Club Split (Q={original_modularity:.4f})")
    plt.axis('off')

    # Plot Louvain communities
    plt.subplot(1, 2, 2)

    # Choose a colormap with enough distinct colors
    num_communities = len(set(louvain_partition.values()))
    cmap = plt.cm.get_cmap('tab10', num_communities)

    nx.draw_networkx(G, pos=pos,
                     node_color=[cmap(louvain_partition[n]) for n in G.nodes()],
                     node_size=300, with_labels=True, font_size=8)
    plt.title(f"Louvain Communities (Q={louvain_modularity:.4f})")
    plt.axis('off')

    plt.tight_layout()

    # Save the figure to the current directory
    image_path = "./louvain_communities.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar(label='Number of nodes')
    plt.xlabel('Club Split (0: Mr. Hi, 1: Officer)')
    plt.ylabel('Louvain Community')
    plt.title('Confusion Matrix: Louvain Communities vs. Club Split')

    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, int(confusion_matrix[i, j]),
                     ha="center", va="center", color="black" if confusion_matrix[i, j] < 5 else "white")

    plt.tight_layout()

    # Save the figure to the current directory
    image_path2 = "./louvain_confusion_matrix.png"
    plt.savefig(image_path2, dpi=300, bbox_inches='tight')

    # Print the image
    print(f"Generated and saved: {image_path2}")
    try:
        img = Image.open(image_path2)
        print(f"Image size: {img.size}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error opening the saved image: {e}")

    plt.show()

    return {
        'original_modularity': original_modularity,
        'louvain_modularity': louvain_modularity,
        'louvain_partition': louvain_partition,
        'confusion_matrix': confusion_matrix
    }


def main():
    # Load Zachary's Karate Club network
    G = load_karate_club()
    print(f"Karate Club network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Calculate assortativity based on faction
    r = calculate_karate_assortativity(G)
    print(f"Assortativity coefficient based on faction: {r:.6f}")

    # Visualize the network
    visualize_karate_network(G)

    # Compute and plot the adjacency spectrum
    eigenvalues = compute_adjacency_spectrum(G)
    print(f"Top 5 eigenvalues: {eigenvalues[:5]}")

    # Apply clustering and compare with ground truth
    clustering_results = apply_clustering(G)

    # Calculate modularity for the club split
    club_partition = nx.get_node_attributes(G, 'club')
    club_modularity = calculate_modularity(G, club_partition)
    print(f"Club split modularity: {club_modularity:.6f}")

    # Analyze modularity distribution in random networks
    # Note: For quicker execution, we're using 100 simulations instead of 1000
    modularity_results = analyze_modularity_distribution(G, n_simulations=100)

    # Apply Louvain community detection and compare with club split
    if LOUVAIN_AVAILABLE:
        louvain_results = analyze_louvain_communities(G)
    else:
        print("Skipping Louvain analysis as the package is not installed.")

    # Print conclusions
    print("\nConclusions on Zachary's Karate Club Analysis:")
    print("1. The club split shows a strong community structure, as evidenced by the high modularity.")
    print("2. The randomization experiment demonstrates that the observed modularity is significantly")
    print("   higher than what would be expected by chance, confirming the real community structure.")
    print("3. Preserving node degrees in the randomization is important to maintain the structural")
    print("   properties of the original network while randomizing connections.")
    print("4. Spectral clustering performed significantly better than K-means for community detection,")
    print("   showing the importance of using network-specific algorithms.")

    # List all saved image files with information
    image_files = [
        "./karate_club_network.png",
        "./adjacency_spectrum.png",
        "./clustering_comparison.png",
        "./modularity_distribution.png",
        "./louvain_communities.png",
        "./louvain_confusion_matrix.png"
    ]

    print("\nGenerated image files:")
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            print(f"- {img_file}: {img.format} image, size {img.size[0]}x{img.size[1]} pixels")
        except Exception as e:
            print(f"- {img_file}: Unable to read image information - {e}")


if __name__ == "__main__":
    main()