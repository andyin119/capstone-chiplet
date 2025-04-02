#!/usr/bin/env python3
import sys
import math
import random
import copy
import json
import argparse
from tqdm import tqdm

# -------------------------------
# Cost functions
# -------------------------------
def cost_crossings(partition, nets):
    """
    For each net, count a cost equal to the number of unique layers minus one.
    This penalizes nets that span multiple layers.
    """
    total = 0
    for net in nets:
        # Only consider cells that exist in the partition.
        layers = { partition[cell] for cell in net if cell in partition }
        if len(layers) > 1:
            total += (len(layers) - 1)
    return total

def cost_area_diff(partition, cells, num_layers):
    """
    Compute the total absolute difference between each layer's area and the average area.
    """
    total_area = sum(cells[cell]['area'] for cell in partition)
    avg_area = total_area / num_layers
    layer_areas = [0.0] * num_layers
    for cell, layer in partition.items():
        layer_areas[layer] += cells[cell]['area']
    return sum(abs(area - avg_area) for area in layer_areas)

def cost_power_diff(partition, cells, num_layers):
    """
    Compute the total absolute difference between each layer's power and the average power.
    Here, we use the "leakage" field as the power consumption.
    """
    total_power = sum(cells[cell]['leakage'] for cell in partition)
    avg_power = total_power / num_layers
    layer_powers = [0.0] * num_layers
    for cell, layer in partition.items():
        layer_powers[layer] += cells[cell]['leakage']
    return sum(abs(power - avg_power) for power in layer_powers)

def total_cost(partition, nets, cells, num_layers, weights):
    """
    Compute a weighted sum of the cost functions.
    weights is a dict with keys 'cross', 'area', 'power'
    """
    c_cross = cost_crossings(partition, nets)
    c_area = cost_area_diff(partition, cells, num_layers)
    c_power = cost_power_diff(partition, cells, num_layers)
    return (weights['cross'] * c_cross +
            weights['area'] * c_area +
            weights['power'] * c_power)

# -------------------------------
# Simulated Annealing Partitioning
# -------------------------------
def simulated_annealing(cells, nets, num_layers, weights, iterations=10000, init_temp=100.0, final_temp=0.1):
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
    return best_partition, best_cost

# -------------------------------
# Input / Output functions
# -------------------------------
def parse_files(hgr_file, json_file):
    """
    Reads nets (edges) from an hgr file and uses a JSON dictionary to look up
    area and power information for each node.

    hgr file: each line formatted as:
         <net> <node1> <node2> ... <node_n>
    The net name (first token) is ignored; the rest are nodes.

    json file: a JSON dictionary mapping node names to a dict containing
               at least "area" and "leakage". If "area" or "leakage" is null,
               they are set to 5 and 0.001 respectively.

    Returns:
      cells: dict mapping node -> {'area': float, 'leakage': float}
             Only nodes present in the hgr file are used.
      nets: list of lists (each inner list is a net consisting of nodes)
    """
    # Build nets and collect nodes from the hgr file.
    nets = []
    nodes_set = set()
    with open(hgr_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            # Ignore the first token (net name)
            net_nodes = tokens[1:]
            if net_nodes:
                nets.append(net_nodes)
                nodes_set.update(net_nodes)
    
    # Load reference dictionary from JSON.
    with open(json_file, 'r') as f:
        ref_dict = json.load(f)
    
    # Build cells dictionary using nodes from the hgr file.
    cells = {}
    for node in nodes_set:
        # Look up node in JSON reference.
        info = ref_dict.get(node, {})
        area = info.get('area', 5.0)
        leakage = info.get('leakage', 0.001)
        if area is None:
            area = 5.0
        if leakage is None:
            leakage = 0.001
        cells[node] = {
            'area': area,
            'leakage': leakage
        }
    
    return cells, nets

def output_results(partition, cells, nets, num_layers, total_cost_value, output_file):
    """
    Writes the partition assignment into the output file.
    It prints the overall cost, per-layer total area and power, and for each layer,
    lists the nodes assigned to that layer.
    """
    # Compute per-layer totals.
    layer_areas = [0.0] * num_layers
    layer_powers = [0.0] * num_layers
    layer_nodes = {l: [] for l in range(num_layers)}
    for cell, layer in partition.items():
        layer_areas[layer] += cells[cell]['area']
        layer_powers[layer] += cells[cell]['leakage']
        layer_nodes[layer].append(cell)
    
    with open(output_file, 'w') as f:
        f.write(f"Total Cost: {total_cost_value:.3f}\n\n")
        for layer in range(num_layers):
            f.write(f"Layer {layer}:\n")
            f.write(f"  Total Area: {layer_areas[layer]:.3f}\n")
            f.write(f"  Total Power: {layer_powers[layer]:.6f}\n")
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
    parser.add_argument("hgr_file", help="Path to the hgr file containing net connectivity.")
    parser.add_argument("json_file", help="Path to the JSON file with cell area and power info.")
    parser.add_argument("-l", "--layers", type=int, default=3,
                        help="Number of layers for partitioning (default: 3).")
    parser.add_argument("-o", "--output", type=str, default="partition_output.txt",
                        help="Output file to dump the partition results (default: partition_output.txt).")
    parser.add_argument("-i", "--iterations", type=int, default=10000,
                        help="Number of iterations for simulated annealing (default: 10000).")
    args = parser.parse_args()

    # Read input files.
    cells, nets = parse_files(args.hgr_file, args.json_file)
    
    # Set weights for the cost functions.
    weights = {
        'cross': 10.0,  # weight for net crossings cost
        'area': 1.0,   # weight for area balance cost
        'power': 1.0   # weight for power balance cost
    }
    
    # Run simulated annealing to obtain the partition.
    best_partition, best_cost = simulated_annealing(
        cells, nets, args.layers, weights, iterations=args.iterations
    )
    
    # Write results to output file.
    output_results(best_partition, cells, nets, args.layers, best_cost, args.output)
    print(f"Partition results written to {args.output}")

if __name__ == "__main__":
    main()
