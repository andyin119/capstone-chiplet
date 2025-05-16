#!/usr/bin/env python3
import sys
import math
import random
import copy
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Constant parameters for power costs
p_edge_cross = 5e-4   # Constant power per edge crossing
p_intra_net  = 3.5e-4  # Constant power per net active in a layer

# Set weights for the cost functions.
# Adjustable for different perference
weights = {
    'cross': 10,  # weight for net crossings cost
    'cross_count': 1, # weight for cost of the number of crossings (penalty)
    'area': 1e-6,    # weight for area balance cost
    'power': 1e5,    # weight for power-related cost
    'mid_pow_penalty': 1e5    # penalty term for reduce power in the middle layer
}

# -------------------------------
# Cost functions
# -------------------------------
def cost_crossings_modified(partition, nets):
    """
    For each net, determine the highest (max) and lowest (min) layer indices of
    its cells, and add a cost equal to the square of the difference.
    This penalizes nets that span multiple layers more heavily.
    """
    total = 0
    for net in nets:
        layers = [partition[cell] for cell in net if cell in partition]
        if layers:
            L_max = max(layers)
            L_min = min(layers)
            diff = L_max - L_min
            total += diff * diff  # quadratic penalty
    return total

def cost_area_diff(partition, cells, num_layers):
    """
    Compute the total absolute difference between each layer's area and the average area.
    """
    total_area = sum(cells[cell]['area'] for cell in partition)
    avg_area = total_area / num_layers
    layer_areas = [0] * num_layers
    for cell, layer in partition.items():
        layer_areas[layer] += cells[cell]['area']
    return sum(abs(area - avg_area) for area in layer_areas)

def cost_power_diff(partition, cells, num_layers):
    """
    Compute the total absolute difference between each layer's total leakage power and the average power.
    Also return the power in the middle layer
    """
    total_power = sum(cells[cell]['total_power'] for cell in partition)
    avg_power = total_power / num_layers
    layer_powers = [0] * num_layers
    for cell, layer in partition.items():
        layer_powers[layer] += cells[cell]['total_power']
    return sum(abs(power - avg_power) for power in layer_powers), layer_powers[1:len(layer_powers) - 1]

def cost_nets_per_layer(partition, nets, num_layers):
    """
    For each layer, count the number of nets active in that layer (i.e. nets that have at least one node assigned).
    Return the total count (across all layers), and the middle layer counts
    """
    layer_net_counts = [0] * num_layers
    for net in nets:
        layers = { partition[cell] for cell in net if cell in partition }
        for l in layers:
            layer_net_counts[l] += 1
    return sum(layer_net_counts), np.array(layer_net_counts[1:len(layer_net_counts) - 1])

def count_edge_crossings(nets, node_to_layer):
    """
    Count the number of edge crossings. For each net, every pair of nodes
    that are assigned to different layers is considered a crossing.
    """
    crossing_count = 0
    for connections in nets:
        # Filter out nodes not assigned in our solution (if any)
        valid_nodes = [node for node in connections if node in node_to_layer]
        # For every unique pair in the net, check if they are in different layers.
        n = len(valid_nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if node_to_layer[valid_nodes[i]] != node_to_layer[valid_nodes[j]]:
                    crossing_count += 1
    return crossing_count

def count_net_crossings(nets, partition):
    """
    Count the number of net crossings. For each hyperedge (net), if not all of its nodes reside
    on a single layer, count that as 1 crossing.
    """
    total = 0
    for net in nets:
        # Only consider cells that exist in the partition.
        layers = { partition[cell] for cell in net if cell in partition }
        if len(layers) > 1:
            total += (len(layers) - 1)
    return total

def total_cost(partition, nets, cells, num_layers, weights):
    """
    Compute a weighted sum of the cost functions.
    The global cost consists of three terms:
      1. Crossing cost, computed via squared difference of layers in nets.
      2. Area balance cost.
      3. Power cost, which includes power balance across layers, a penalty for edge crossings (multiplied by p_edge_cross),
         and a penalty for the number of nets active per layer (multiplied by p_intra_net).
         
    weights is a dict with keys 'cross', 'area', 'power'
    """
    # Edge crossing cost (penalizes nets spanning multiple layers)
    c_cross = cost_crossings_modified(partition, nets)
    # Area balancing cost
    c_area = cost_area_diff(partition, cells, num_layers)
    # Power imbalance (difference between each layer's total power and the average power)
    c_power, c_mid_power = cost_power_diff(partition, cells, num_layers)
    # Additional power cost from edge crossings and nets active in layers
    c_edge_power = p_edge_cross * c_cross
    all_net_counts, mid_net_counts = cost_nets_per_layer(partition, nets, num_layers)
    c_net_layer  = p_intra_net * all_net_counts
    c_net_mid_layer = p_intra_net * mid_net_counts
    total_edge_crossing = count_edge_crossings(nets, partition)
    # print(c_cross)
    # print(total_edge_crossing)
    # print(c_area)
    # print(c_power + c_edge_power + c_net_layer)
    # print(sum(c_mid_power) / len(c_mid_power) + sum(c_net_mid_layer) / len(c_net_mid_layer))
    total = (weights['cross'] * c_cross + 
             weights['cross_count'] * total_edge_crossing + 
             weights['area'] * c_area +
             weights['power'] * (c_power + c_edge_power + c_net_layer) + 
             weights['mid_pow_penalty'] * (sum(c_mid_power) / len(c_mid_power) + sum(c_net_mid_layer) / len(c_net_mid_layer)))
    return total

# -------------------------------
# Simulated Annealing Partitioning
# -------------------------------
def simulated_annealing(cells, nets, num_layers, weights, iterations=100000, init_temp=500.0, final_temp=0.1):
    # Start with a balanced partition.
    partition = {}
    cell_list = list(cells.keys())
    # Assign cells to layers in round-robin fashion.
    for i, cell in enumerate(cell_list):
        partition[cell] = i % num_layers
    random.shuffle(cell_list)
    
    current_cost = total_cost(partition, nets, cells, num_layers, weights)
    best_partition = copy.deepcopy(partition)
    best_cost = current_cost

    cost_history = [float(current_cost)]
    for i in tqdm(range(iterations)):
        # Exponential decay temperature schedule.
        T = init_temp * (final_temp / init_temp) ** (i / iterations)

        # Pick a random cell and assign it to a different random layer.
        cell_to_change = random.choice(cell_list)
        current_layer = partition[cell_to_change]
        new_layer = random.choice([l for l in range(num_layers) if l != current_layer])
        new_partition = copy.deepcopy(partition)
        new_partition[cell_to_change] = new_layer
        
        new_cost = total_cost(new_partition, nets, cells, num_layers, weights)
        delta = new_cost - current_cost
        
        # Metropolis criterion.
        if delta < 0 or random.random() < math.exp(-delta / T):
            partition = new_partition
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_partition = copy.deepcopy(new_partition)
        cost_history.append(float(current_cost))
    return best_partition, best_cost, cost_history

# -------------------------------
# Input / Output functions
# -------------------------------
def parse_files(nets_file, json_file):
    """
    Reads nets (edges) from a JSON file and uses a second JSON dictionary file to look up
    area and power information for each node.

    nets_file: a JSON file with a list of dictionaries.
               Each dictionary must have:
                 - "net": a name for the net (ignored)
                 - "connections": a list of nodes forming that net.
    json_file: a JSON dictionary mapping node names to a dict containing
               at least "area" and "leakage". If "area" or "leakage" is null,
               they are set to 5 and 0.001 respectively.

    Returns:
      cells: dict mapping node -> {'area': float, 'total_power': float}
             Only nodes present in the nets_file are used.
      nets: list of lists (each inner list is a net consisting of nodes)
    """
    nets = []
    nodes_set = set()

    # Load nets from JSON file.
    with open(nets_file, 'r') as f:
        nets_data = json.load(f)
        for net_info in nets_data:
            connections = net_info.get("connections", [])
            if connections:
                nets.append(connections)
                nodes_set.update(connections)
    
    # Load reference dictionary from JSON.
    with open(json_file, 'r') as f:
        ref_dict = json.load(f)
    
    cells = {}
    for node in ref_dict:
        name = node["node"]
        # Use default values if None.
        area = node.get("area")
        if area is None:
            area = 5.0
        power = node.get("total_power")
        if power is None:
            power = 0.001
        cells[name] = {'area': area, 'total_power': power}
    
    return cells, nets

def output_results(partition, cells, nets, num_layers, total_cost_value, output_file, edge_crossings, edge_crossings_v2):
    """
    Writes the partition assignment into the output file.
    It prints the overall cost, per-layer total area and power, and for each layer,
    lists the nodes assigned to that layer.
    """
    # Compute per-layer totals.
    layer_areas = [0] * num_layers
    layer_powers = [0] * num_layers
    layer_nodes = {l: [] for l in range(num_layers)}
    net_powers = [0] * num_layers
    for cell, layer in partition.items():
        layer_areas[layer] += cells[cell]['area']
        layer_powers[layer] += cells[cell]['total_power']
        layer_nodes[layer].append(cell)
    
    for layer in range(num_layers):
        intra_count = 0
        node_list = [cell for cell, l in partition.items() if l == layer]
        for connections in nets:
            # accumulate the number of nodes to approximate the net counts
            for node in connections:
                if node in node_list:
                    intra_count += 1
        net_powers[layer] += intra_count * p_intra_net

    with open(output_file, 'w') as f:
        f.write(f"Total Cost: {total_cost_value:.3f}\n")
        f.write(f"Total Edge Crossings: {edge_crossings}\n")
        f.write(f"Total Edge Crossings v2: {edge_crossings_v2}\n")
        f.write(f"Total Edge Crossings Power: {edge_crossings_v2 * p_edge_cross:.15f}\n\n")
        for layer in range(num_layers):
            f.write(f"Layer {layer}:\n")
            f.write(f"  Total Area: {layer_areas[layer]:.3f}\n")
            f.write(f"  Total Cell Power: {layer_powers[layer]:.6f}\n")
            f.write(f"  Total Net Power: {net_powers[layer]:.15f}\n")
            f.write(f"  Node Count: {len(layer_nodes[layer])}\n")
            f.write("  Nodes:\n")
            for node in sorted(layer_nodes[layer]):
                f.write(f"    {node}\n")
            f.write("\n")

# -------------------------------
# Main function with argparse
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multiobjective partitioning using simulated annealing."
    )
    parser.add_argument("nets_file", help="Path to the nets JSON file containing connectivity (e.g. nets.json).")
    parser.add_argument("json_file", help="Path to the JSON file with cell area and power info.")
    parser.add_argument("-l", "--layers", type=int, default=3,
                        help="Number of layers for partitioning (default: 3).")
    parser.add_argument("-o", "--output", type=str, default="partition_output.txt",
                        help="Output file to dump the partition results (default: partition_output.txt).")
    parser.add_argument("-i", "--iterations", type=int, default=10000,
                        help="Number of iterations for simulated annealing (default: 10000).")
    parser.add_argument("-t_i", "--initial_temp", type=float, default=500,
                        help="Initial Temperature of the simulated annealing algorithm")
    parser.add_argument("-t_f", "--final_temp", type=float, default=0.1,
                        help="Final Temperature of the simulated annealing algorithm")
    parser.add_argument("--plot_cost", action="store_true",
                        help="Plot the cost trend of the simulated annealing algorithm")
    args = parser.parse_args()

    # Read input files.
    cells, nets = parse_files(args.nets_file, args.json_file)
    
    total_area = 0
    total_power = 0
    for key, value in cells.items():
        total_area += value['area']
        total_power += value['total_power']
    print(total_area)
    print(total_power)
    
    # Run simulated annealing to obtain the partition.
    best_partition, best_cost, cost_history = simulated_annealing(
        cells, nets, args.layers, weights, iterations=args.iterations,
        init_temp=args.inital_temp, final_temp=args.final_temp
    )
    
    # Count the number of edge crossings (a crossing is counted for each unique pair of nodes in the same net that are in different layers)
    edge_crossings = count_edge_crossings(nets, best_partition)

    net_crossings = count_net_crossings(nets, best_partition)
    
    # Write results to output file.
    output_results(best_partition, cells, nets, args.layers, best_cost, args.output, edge_crossings, net_crossings)
    print(f"Partition results written to {args.output}")

    if args.plot_cost:
        plt.figure()
        plt.plot(cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Simulated Annealing Cost Trend')
        plt.show()

if __name__ == "__main__":
    main()
