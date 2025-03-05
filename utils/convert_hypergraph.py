#!/usr/bin/env python3
import argparse
import math
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

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
      
    This function reads both the soft blocks and terminal definitions and returns a dictionary of nodes.
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
            line_idx += 1  # advance past the counts
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
    Since the nets file does not provide explicit names for the nets, we generate names automatically.
    """
    hyperedges = {}
    with open(file_path, 'r') as f:
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
            # For every pair in this hyperedge, add an edge labeled with the net id
            for u, v in combinations(unique_nodes, 2):
                G.add_edge(u, v, net=net_id)
    return G

def visualize_multigraph(G):
    """
    Visualizes the multigraph produced by the clique expansion.
    
    Node sizes are scaled based on the area (if available) so that larger blocks appear bigger.
    For nodes without an area (like terminals), a default size is used.
    """
    # Compute node sizes: use sqrt(area) for blocks with area, else a default size.
    default_size = 300  # default node size for terminals or missing area
    sizes = {}
    for node, data in G.nodes(data=True):
        if "area" in data and data["area"] is not None:
            # Scale using square root; adjust multiplier as needed for your visualization.
            sizes[node] = math.sqrt(data["area"]) * 2
        else:
            sizes[node] = default_size
    # Create a list of sizes in the same order as G.nodes
    node_size_list = [sizes[n] for n in G.nodes()]
    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))
    
    # Draw nodes with size based on the area
    nx.draw_networkx_nodes(G, pos, node_size=node_size_list, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
    
    # Draw edge labels showing the net id
    edge_labels = {(u, v): data['net'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title("Hypergraph Visualization via Clique Expansion\n(Node size scaled by block area)")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Convert Bookshelf blocks and nets files into a hypergraph (blocks as nodes, nets as hyperedges) and visualize it with block sizes."
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

    # Convert the hypergraph into a multigraph by clique expansion
    G = hypergraph_to_multigraph(nodes, hyperedges)
    visualize_multigraph(G)

if __name__ == "__main__":
    main()
