#!/usr/bin/env python3
import argparse
import math
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def parse_blocks(file_path):
    """
    Parses a Bookshelf blocks file.
    
    Expected format (from your n10.blocks.txt):
      UCSC blocks 1.0
      # Created      : Fri Dec 08 19:15:43 PST 2000
      ...
      NumSoftRectangularBlocks : 10
      NumHardRectilinearBlocks : 0
      NumTerminals : 69
      
      sb0 softrectangular 16318 0.300 3.000
      sb1 softrectangular 24045 0.300 3.000
      ...
      p1 terminal
      p2 terminal
      ...
      
    Returns a dictionary mapping block/terminal names to attributes.
    """
    nodes = {}
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Read header counts
    soft_blocks = 0
    hard_blocks = 0
    terminals = 0
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith("NumSoftRectangularBlocks"):
            parts = line.split(":")
            if len(parts) > 1:
                soft_blocks = int(parts[1].strip())
        elif line.startswith("NumHardRectilinearBlocks"):
            parts = line.split(":")
            if len(parts) > 1:
                hard_blocks = int(parts[1].strip())
        elif line.startswith("NumTerminals"):
            parts = line.split(":")
            if len(parts) > 1:
                terminals = int(parts[1].strip())
            line_idx += 1  # move past the counts line
            break
        line_idx += 1

    # Read soft rectangular blocks
    for i in range(soft_blocks):
        if line_idx >= len(lines):
            break
        tokens = lines[line_idx].split()
        # Expected: <name> softrectangular <area> <min_aspect> <max_aspect>
        if len(tokens) >= 5:
            name = tokens[0]
            block_type = tokens[1]
            try:
                area = float(tokens[2])
            except ValueError:
                area = None
            try:
                min_aspect = float(tokens[3])
            except ValueError:
                min_aspect = None
            try:
                max_aspect = float(tokens[4])
            except ValueError:
                max_aspect = None
            nodes[name] = {
                "type": block_type,
                "area": area,
                "min_aspect": min_aspect,
                "max_aspect": max_aspect
            }
        line_idx += 1

    # Read terminals
    for i in range(terminals):
        if line_idx >= len(lines):
            break
        tokens = lines[line_idx].split()
        # Expected: <name> terminal
        if len(tokens) >= 2:
            name = tokens[0]
            nodes[name] = {"type": tokens[1]}
        line_idx += 1

    return nodes

def parse_nets(file_path):
    """
    Parses a Bookshelf nets file.
    
    Expected format (from your n10.nets.txt):
      UCLA nets 1.0
      # Created      : 2000
      ...
      NumNets : 118
      NumPins : 248
      NetDegree : 2
      p1 B
      sb6 B
      NetDegree : 2
      p2 B
      sb8 B
      ...
      
    Each net is defined by a "NetDegree" line that tells how many pins (node connections) follow.
    Since the nets file does not provide explicit net names, we generate them automatically.
    Returns a dictionary mapping net names to the list of connected block names.
    """
    hyperedges = {}
    with open(file_path, 'r') as f:
        # Skip header lines starting with UCLA or #
        lines = [line.strip() for line in f if line.strip() and not line.startswith("UCLA") and not line.startswith("#")]
    
    net_counter = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("NumNets") or line.startswith("NumPins"):
            i += 1
            continue
        if line.startswith("NetDegree"):
            parts = line.split(":")
            if len(parts) < 2:
                i += 1
                continue
            try:
                degree = int(parts[1].strip())
            except ValueError:
                degree = 0
            net_name = f"net{net_counter}"
            net_counter += 1
            hyperedges[net_name] = []
            # Next 'degree' lines list the pins (each line: <node> B)
            for j in range(degree):
                if i + 1 + j >= len(lines):
                    break
                pin_tokens = lines[i + 1 + j].split()
                if len(pin_tokens) >= 1:
                    node_name = pin_tokens[0]
                    hyperedges[net_name].append(node_name)
            i += 1 + degree
        else:
            i += 1
    return hyperedges

def hypergraph_to_multigraph(nodes, hyperedges):
    """
    Converts the hypergraph into a multigraph via clique expansion.
    
    - Nodes: the blocks/terminals from the blocks file.
    - For each hyperedge (net) connecting two or more nodes, add an edge between every pair
      of nodes with an attribute 'net' indicating the originating hyperedge.
    """
    G = nx.MultiGraph()
    # Add all nodes (blocks)
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)
    
    for net_id, node_list in hyperedges.items():
        unique_nodes = list(set(node_list))
        if len(unique_nodes) > 1:
            for u, v in combinations(unique_nodes, 2):
                G.add_edge(u, v, net=net_id)
    return G

def compute_partitions(G):
    """
    Partitions the graph using a modularity-based community detection.
    Here we use NetworkX's greedy_modularity_communities to compute communities.
    Returns a dictionary mapping node to partition index and the communities list.
    """
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    partition_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            partition_map[node] = idx
    return partition_map, communities

def visualize_partitioned_graph(G, partition_map):
    """
    Visualizes the multigraph with node sizes scaled by block area and colors 
    determined by the partition assignment.
    """
    # Compute node sizes: if a node has an "area", scale using sqrt(area), else use default.
    default_size = 300  # default node size for terminals or nodes missing area
    sizes = {}
    for node, data in G.nodes(data=True):
        if "area" in data and data["area"] is not None:
            sizes[node] = math.sqrt(data["area"]) * 2
        else:
            sizes[node] = default_size
    node_size_list = [sizes[n] for n in G.nodes()]
    
    # Create a color map for partitions.
    # We'll use a matplotlib colormap with enough distinct colors.
    num_partitions = max(partition_map.values()) + 1 if partition_map else 1
    cmap = plt.cm.get_cmap('tab20', num_partitions)
    node_colors = [cmap(partition_map.get(n, 0)) for n in G.nodes()]
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
    
    # Draw edge labels showing the net id (optional)
    edge_labels = {(u, v): data['net'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Graph Partitioning via Clique Expansion\n(Node sizes scaled by block area, colored by partition)")
    plt.axis('off')
    plt.show()
    
    # Print out partition summary
    partitions = {}
    for node, part in partition_map.items():
        partitions.setdefault(part, []).append(node)
    for part, nodes_list in partitions.items():
        print(f"Partition {part}: {nodes_list}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Bookshelf blocks and nets files into a hypergraph (blocks as nodes, nets as hyperedges), find graph partitions, and visualize the result."
    )
    parser.add_argument("--blocks", required=True,
                        help="Path to the Bookshelf blocks file (e.g., n10.blocks.txt)")
    parser.add_argument("--nets", required=True,
                        help="Path to the Bookshelf nets file (e.g., n10.nets.txt)")
    args = parser.parse_args()

    nodes = parse_blocks(args.blocks)
    hyperedges = parse_nets(args.nets)
    
    print("Parsed Nodes:")
    for node, attrs in nodes.items():
        print(f"  {node}: {attrs}")
    
    print("\nParsed Hyperedges (Nets):")
    for net, node_list in hyperedges.items():
        print(f"  {net}: {node_list}")

    # Create the multigraph using clique expansion
    G = hypergraph_to_multigraph(nodes, hyperedges)
    
    # Compute partitions using greedy modularity communities
    partition_map, communities = compute_partitions(G)
    
    # Visualize the graph with nodes colored by their partition
    visualize_partitioned_graph(G, partition_map)

if __name__ == "__main__":
    main()
