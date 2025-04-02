# #!/usr/bin/env python3
# import argparse
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# def parse_layer_file(layer_filename):
#     """
#     Parse the layer file (like moo_toy_result.txt) to obtain a dictionary mapping each node to its layer.
#     Assumes the file format as in the sample:
    
#       Total Cost: ...
      
#       Layer 0:
#         Total Area: ...
#         Total Power: ...
#         Nodes:
#           <node1>
#           <node2>
#           ...
      
#       Layer 1:
#       ...
    
#     Returns:
#       layer_dict: dict mapping node -> layer index (int)
#       layers: dict mapping layer index -> list of nodes in that layer
#     """
#     layer_dict = {}
#     layers = {}
#     current_layer = None
#     with open(layer_filename, 'r') as f:
#         for line in f:
#             line = line.rstrip()
#             if line.startswith("Layer "):
#                 # Expect format: "Layer <n>:" 
#                 parts = line.split()
#                 try:
#                     current_layer = int(parts[1].rstrip(':'))
#                     layers[current_layer] = []
#                 except ValueError:
#                     continue
#             elif current_layer is not None and line.strip().startswith("\\") or (current_layer is not None and not line.strip().startswith("Total") and not line.strip().startswith("Nodes:") and line.strip()):
#                 # If the line is indented (or not one of the header lines) assume it is a node.
#                 node = line.strip()
#                 # Skip if it's not a node (e.g. "Nodes:" header)
#                 if node == "Nodes:":
#                     continue
#                 layers[current_layer].append(node)
#                 layer_dict[node] = current_layer
#     return layer_dict, layers

# def parse_hgr_file(hgr_filename):
#     """
#     Parse the hgr file where each line is:
#       <net> <node1> <node2> ... <node_n>
#     Returns a list of hyperedges, each hyperedge is a list of nodes.
#     """
#     hyperedges = []
#     with open(hgr_filename, 'r') as f:
#         for line in f:
#             tokens = line.strip().split()
#             if not tokens or len(tokens) < 2:
#                 continue
#             # Ignore the first token (net name) and keep the rest as nodes
#             hyperedges.append(tokens[1:])
#     return hyperedges

# def compute_layout(layers, x_spacing=2.0, y_spacing=2.0):
#     """
#     Compute (x,y) positions for all nodes based on their layer.
#     layers: dict mapping layer index -> list of nodes.
#     Returns a dict mapping node -> (x,y)
#     """
#     pos = {}
#     # Determine the number of layers and assign a y coordinate for each layer.
#     sorted_layers = sorted(layers.keys())
#     for layer in sorted_layers:
#         nodes = layers[layer]
#         n = len(nodes)
#         # even spacing for nodes in this layer.
#         # For example, center the nodes horizontally around 0.
#         if n > 1:
#             xs = np.linspace(-((n-1)/2)*x_spacing, ((n-1)/2)*x_spacing, n)
#         else:
#             xs = [0]
#         y = -layer * y_spacing  # layer 0 at top; increasing layer means going downward.
#         for i, node in enumerate(nodes):
#             pos[node] = (xs[i], y)
#     return pos

# def draw_hypergraph(pos, hyperedges, layer_dict, output_file):
#     """
#     Draw the hypergraph using matplotlib.
#     - pos: dictionary mapping node -> (x,y)
#     - hyperedges: list of hyperedges (each hyperedge is a list of nodes)
#     - layer_dict: mapping node -> layer index (for color coding)
#     """
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # Define a color map for layers.
#     layer_colors = {}
#     unique_layers = sorted(set(layer_dict.values()))
#     cmap = plt.get_cmap("tab10")
#     for i, layer in enumerate(unique_layers):
#         layer_colors[layer] = cmap(i % 10)
    
#     # Draw nodes.
#     for node, (x,y) in pos.items():
#         layer = layer_dict.get(node, -1)
#         color = layer_colors.get(layer, "gray")
#         ax.plot(x, y, 'o', color=color, markersize=8)
#         #ax.text(x, y+0.2, node, fontsize=8, ha='center', va='bottom', rotation=0)
    
#     # For each hyperedge, if it connects 2 or more nodes, compute the barycenter and draw connecting lines.
#     for edge in hyperedges:
#         # Filter out nodes not in pos.
#         nodes_in_edge = [node for node in edge if node in pos]
#         if len(nodes_in_edge) < 2:
#             continue
#         # Get positions.
#         pts = np.array([pos[node] for node in nodes_in_edge])
#         center = pts.mean(axis=0)
#         # Optionally draw a small marker at the center.
#         ax.plot(center[0], center[1], 'kx', markersize=6, alpha=0.5)
#         # Draw lines from the center to each node.
#         for node in nodes_in_edge:
#             x,y = pos[node]
#             ax.plot([center[0], x], [center[1], y], '-', color='gray', alpha=0.5, linewidth=1)
    
#     # Draw horizontal lines to separate layers (optional)
#     for layer in unique_layers:
#         y = -layer * 2.0  # same as y_spacing used in compute_layout
#         ax.axhline(y=y, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
#     ax.set_title("Hypergraph Visualization by Layers")
#     ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300)
#     print(f"Visualization saved as {output_file}")

# def main():
#     parser = argparse.ArgumentParser(
#         description="Visualize a hypergraph with nodes separated into layers and hyperedges drawn as common nets."
#     )
#     parser.add_argument("hgr_file", help="Path to the hgr file containing hyperedge connectivity.")
#     parser.add_argument("layer_file", help="Path to the layer file (e.g. moo_toy_result.txt) with node layer assignments.")
#     parser.add_argument("-o", "--output", type=str, default="hypergraph_vis.png",
#                         help="Output image file (default: hypergraph_vis.png)")
#     args = parser.parse_args()
    
#     # Parse input files.
#     hyperedges = parse_hgr_file(args.hgr_file)
#     layer_dict, layers = parse_layer_file(args.layer_file)
    
#     # Compute layout for nodes.
#     pos = compute_layout(layers, x_spacing=2.0, y_spacing=2.0)
    
#     # Draw and save the hypergraph visualization.
#     draw_hypergraph(pos, hyperedges, layer_dict, args.output)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def parse_layer_file(layer_filename):
    """
    Parse the layer file (like moo_toy_result.txt) to obtain:
      - layer_dict: mapping each node to its layer index (int)
      - layers: dictionary mapping layer index -> list of nodes in that layer.
    
    Assumes the file format as in your sample.
    """
    layer_dict = {}
    layers = {}
    current_layer = None
    with open(layer_filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Layer "):
                # Expect format: "Layer <n>:" 
                parts = line.split()
                try:
                    current_layer = int(parts[1].rstrip(':'))
                    layers[current_layer] = []
                except ValueError:
                    continue
            elif current_layer is not None:
                # Look for lines with node names (skip headers like "Total Area:" etc.)
                stripped = line.strip()
                if stripped and stripped not in {"Nodes:", "Total Cost:"} and not stripped.startswith("Total"):
                    layers[current_layer].append(stripped)
                    layer_dict[stripped] = current_layer
    return layer_dict, layers

def parse_hgr_file(hgr_filename):
    """
    Parse the hgr file where each line is:
      <net> <node1> <node2> ... <node_n>
    Returns a list of hyperedges, where each hyperedge is a list of nodes.
    """
    hyperedges = []
    with open(hgr_filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or len(tokens) < 2:
                continue
            # Ignore the first token (net name) and keep the rest as nodes.
            hyperedges.append(tokens[1:])
    return hyperedges

def compute_global_layout(layers, x_spacing=2.0, y_spacing=2.0):
    """
    Compute (x,y) positions for all nodes based on their layer.
    layers: dict mapping layer index -> list of nodes.
    Returns a dict mapping node -> (x,y)
    """
    pos = {}
    sorted_layers = sorted(layers.keys())
    for layer in sorted_layers:
        nodes = layers[layer]
        n = len(nodes)
        if n > 1:
            xs = np.linspace(-((n-1)/2)*x_spacing, ((n-1)/2)*x_spacing, n)
        else:
            xs = [0]
        y = -layer * y_spacing  # layer 0 at top; increasing layer means going downward.
        for i, node in enumerate(nodes):
            pos[node] = (xs[i], y)
    return pos

def draw_overall_hypergraph(pos, hyperedges, layer_dict, output_file):
    """
    Draw the overall hypergraph:
      - Nodes are plotted at positions given by pos.
      - Each hyperedge (with at least two nodes) is drawn by computing the barycenter
        of the nodes in that hyperedge and drawing lines from that center to each node.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Assign colors for layers.
    unique_layers = sorted(set(layer_dict.values()))
    cmap = plt.get_cmap("tab10")
    layer_colors = {layer: cmap(i % 10) for i, layer in enumerate(unique_layers)}
    
    # Draw nodes.
    for node, (x,y) in pos.items():
        layer = layer_dict.get(node, -1)
        color = layer_colors.get(layer, "gray")
        ax.plot(x, y, 'o', color=color, markersize=8)
        #ax.text(x, y+0.2, node, fontsize=8, ha='center', va='bottom', rotation=0)
    
    # Draw hyperedges.
    for edge in hyperedges:
        # Filter out nodes not in pos.
        nodes_in_edge = [node for node in edge if node in pos]
        if len(nodes_in_edge) < 2:
            continue
        pts = np.array([pos[node] for node in nodes_in_edge])
        center = pts.mean(axis=0)
        # Draw a marker at the center.
        ax.plot(center[0], center[1], 'kx', markersize=6, alpha=0.5)
        # Draw a line from the center to each node.
        for node in nodes_in_edge:
            x,y = pos[node]
            ax.plot([center[0], x], [center[1], y], '-', color='gray', alpha=0.5, linewidth=1)
    
    # Optionally, draw horizontal lines separating layers.
    for layer in unique_layers:
        y = -layer * 2.0  # assuming y_spacing=2.0
        ax.axhline(y=y, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.set_title("Overall Hypergraph Visualization")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Overall hypergraph saved as {output_file}")

def draw_intralayer_graph(layers, hyperedges, output_file):
    """
    For each layer, draw a subplot showing only intra-layer hyperedges.
    For each layer, nodes are arranged evenly along a horizontal line and hyperedges
    that connect only nodes in that layer are drawn.
    """
    num_layers = len(layers)
    sorted_layer_ids = sorted(layers.keys())
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4*num_layers), sharex=False)
    if num_layers == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, sorted_layer_ids):
        nodes = layers[layer]
        n = len(nodes)
        if n > 1:
            xs = np.linspace(-((n-1)/2)*2.0, ((n-1)/2)*2.0, n)
        else:
            xs = [0]
        pos_layer = {node: (xs[i], 0) for i, node in enumerate(nodes)}
        
        # Draw nodes.
        for node, (x,y) in pos_layer.items():
            ax.plot(x, y, 'o', color='tab:blue', markersize=8)
            ax.text(x, y+0.2, node, fontsize=8, ha='center', va='bottom', rotation=0)
        
        # For each hyperedge, if all its nodes are in this layer, draw the connection.
        nodes_set = set(nodes)
        for edge in hyperedges:
            # Only consider hyperedges that are fully contained in this layer.
            if set(edge).issubset(nodes_set) and len(edge) >= 2:
                pts = np.array([pos_layer[node] for node in edge])
                center = pts.mean(axis=0)
                ax.plot(center[0], center[1], 'kx', markersize=6, alpha=0.5)
                for node in edge:
                    x,y = pos_layer[node]
                    ax.plot([center[0], x], [center[1], y], '-', color='gray', alpha=0.5, linewidth=1)
        
        ax.set_title(f"Layer {layer} Intra-layer Connections")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Intralayer graph saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a hypergraph with nodes arranged by layers and display both overall and intra-layer connections."
    )
    parser.add_argument("hgr_file", help="Path to the hgr file containing hyperedge connectivity.")
    parser.add_argument("layer_file", help="Path to the layer file with node layer assignments.")
    parser.add_argument("-o", "--output", type=str, default="hypergraph_vis.png",
                        help="Output image file for the overall hypergraph (default: hypergraph_vis.png)")
    parser.add_argument("--intra_output", type=str, default="hypergraph_intralayer.png",
                        help="Output image file for the intra-layer graph (default: hypergraph_intralayer.png)")
    args = parser.parse_args()
    
    # Parse input files.
    hyperedges = parse_hgr_file(args.hgr_file)
    layer_dict, layers = parse_layer_file(args.layer_file)
    
    # Compute global layout for overall hypergraph.
    pos = compute_global_layout(layers, x_spacing=2.0, y_spacing=2.0)
    
    # Draw and save the overall hypergraph.
    draw_overall_hypergraph(pos, hyperedges, layer_dict, args.output)
    
    # Draw and save the intra-layer connections.
    draw_intralayer_graph(layers, hyperedges, args.intra_output)

if __name__ == "__main__":
    main()

