import json
import argparse
import cvxpy as cp
import numpy as np
from tqdm import tqdm

# Constants for global power cost:
p_edge_cross = 5e-4   # constant power per edge crossing
p_intra_net = 3.5e-4   # constant power per net within a layer

def load_data(gates_file, nets_file):
    with open(gates_file, 'r') as f:
        data = json.load(f)
    # Convert list of nodes to a dictionary: key=node name, value=attributes
    gates = {node["node"]: {k: v for k, v in node.items() if k != "node"}
             for node in data}
    with open(nets_file, 'r') as f:
        nets = json.load(f)
    return gates, nets

def get_upper_triangle_mask(d):
    """Return a (d x d) mask with ones above the main diagonal."""
    return np.triu(np.ones((d, d)), k=1)

def build_edge_cost(L, net_indices_list, lambda_edge):
    """
    Computes the edge crossing penalty in a vectorized way over all nets.
    - L is the CVXPY expression for layer assignments (a vector, length = num_nodes).
    - net_indices_list is a list of lists, where each inner list contains the indices
      of the nodes participating in that net.
    """
    f_edge = 0
    # Cache for upper-triangular masks, keyed by net size
    mask_cache = {}
    for valid_indices in net_indices_list:
        d = len(valid_indices)
        if d < 2:
            continue
        if d not in mask_cache:
            mask_cache[d] = get_upper_triangle_mask(d)
        mask = mask_cache[d]
        
        # Gather the layer values for nodes in this net
        L_net = cp.hstack([L[i] for i in valid_indices])
        # Reshape into a column vector (d x 1)
        L_net_col = cp.reshape(L_net, (d, 1))
        # Transpose to obtain a row vector (1 x d)
        L_net_row = cp.transpose(L_net_col)
        # Compute the matrix of squared differences using broadcasting.
        diff_matrix = cp.square(L_net_col - L_net_row)
        # Sum over all unique pairs (using the precomputed upper-triangular mask)
        net_penalty = cp.sum(cp.multiply(mask, diff_matrix))
        f_edge += net_penalty
    return lambda_edge * f_edge

def build_problem(gates, nets, num_layers, lambda_edge=1.0, lambda_power=1.0, lambda_area=1.0, p_edge_cross=0.0005, p_intra_net=0.000005):
    # List all nodes and create an index mapping
    nodes = list(gates.keys())
    num_nodes = len(nodes)
    print("Number of nodes:", num_nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Extract area and leakage (power) information
    areas = np.array([gates[node]['area'] for node in nodes])
    leakages = np.array([gates[node]['total_power'] for node in nodes])
    total_area = np.sum(areas)
    total_power = np.sum(leakages)
    
    # Decision variables: x[i, p] indicates the (soft) assignment of node i to layer p.
    # Note: We relax x to be in [0, 1] with the constraint sum_p x[i,p] = 1.
    x = cp.Variable((num_nodes, num_layers))
    
    # Each node must be assigned to one layer.
    constraints = [cp.sum(x[i, :]) == 1 for i in range(num_nodes)]
    constraints += [x >= 0, x <= 1]
    
    # Compute layer assignment L[i] = sum_{p} p * x[i, p]
    layer_values = np.arange(num_layers, dtype=float)  # e.g., [0, 1, 2, ...]
    L = x @ layer_values

    # Precompute Net Indices for all nets
    net_indices_list = []
    for net in nets:
        connections = net["connections"]
        valid_indices = [node_index[node] for node in connections if node in node_index]
        if len(valid_indices) >= 2:
            net_indices_list.append(valid_indices)
    
    # Edge Crossing Cost and Global Power from Edge Crossings
    f_edge = build_edge_cost(L, net_indices_list, lambda_edge)
    # Use the same term for global power cost due to crossings.
    f_power_edge = p_edge_cross * f_edge

    # Intra-layer Net Power Cost
    f_intra_net = 0
    for net in tqdm(nets, desc="Processing nets for intra-layer net power cost"):
        connections = net["connections"]
        valid_indices = [node_index[node] for node in connections if node in node_index]
        n_net = len(valid_indices)
        if n_net == 0:
            continue
        for p in range(num_layers):
            S = cp.sum(cp.hstack([x[idx, p] for idx in valid_indices]))
            f_intra_net += cp.square(S) / n_net
    f_intra_net = p_intra_net * f_intra_net

    # Area Balancing Cost
    target_area = total_area / num_layers
    f_area = 0
    for p in range(num_layers):
        area_p = cp.sum(cp.multiply(areas, x[:, p]))
        f_area += cp.square(area_p - target_area)
    f_area = lambda_area * f_area
    
    # Overall Objective
    # Combine edge cost, area balancing cost, and global power cost.
    objective = cp.Minimize(f_edge + f_area +
                            lambda_power * (f_power_edge + f_intra_net))
    
    # Power Capacity Constraint for Middle Layers
    for p in range(1, num_layers - 1):
        constraints.append(cp.sum(cp.multiply(leakages, x[:, p])) <= total_power / num_layers)
    
    problem = cp.Problem(objective, constraints)
    return problem, x, nodes

def get_partitioning(x_opt, nodes, num_layers):
    """Extracts a hard partitioning from the solution x_opt."""
    partitions = {p: [] for p in range(num_layers)}
    x_val = x_opt.value
    for i, node in enumerate(nodes):
        # Hard assignment: choose the layer with the highest value
        p = int(np.argmax(x_val[i, :]))
        partitions[p].append(node)
    return partitions

def compute_partition_stats(partitions, gates, nets, p_intra_net):
    """
    Compute statistics for each partition, including total area, total cell power,
    and intra-net power (power due to nets fully contained within the partition).
    """
    stats = {}
    # For each layer/partition p
    for p, node_list in partitions.items():
        # Sum area and cell power
        total_area = sum(gates[node]['area'] for node in node_list if node in gates)
        total_cell_power = sum(gates[node]['total_power'] for node in node_list if node in gates)

        # Count nets fully contained within this partition
        intra_count = 0
        for net in nets:
            # accumulate the number of nodes to approximate the net counts
            for node in net['connections']:
                if node in node_list:
                    intra_count += 1

        # Compute intra-net power for this partition
        intra_net_power = intra_count * p_intra_net

        # Collect stats
        stats[p] = {
            "nodes": node_list,
            "node_size": len(node_list), # number of nodes in this layer
            "total_area": total_area,
            "total_cell_power": total_cell_power,
            "intra_net_power": intra_net_power,
            "total_layer_power": total_cell_power + intra_net_power
        }
    return stats

def count_edge_crossings(nets, node_to_layer):
    """
    Count the number of edge crossings. For each net, every pair of nodes
    that are assigned to different layers is considered a crossing.
    """
    crossing_count = 0
    for net in nets:
        connections = net["connections"]
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
        connections = net["connections"]
        # Only consider cells that exist in the partition.
        layers = { partition[cell] for cell in connections if cell in partition }
        if len(layers) > 1:
            total += (len(layers) - 1)
    return total

def main():
    parser = argparse.ArgumentParser(description="Optimized Convex QP for 3D Chiplet Circuit Partitioning")
    parser.add_argument("--gates", required=True, help="Path to gates.json")
    parser.add_argument("--nets", required=True, help="Path to nets.json")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers (partitions)")
    parser.add_argument("--lambda_edge", type=float, default=100, help="Weight for edge crossing cost")
    parser.add_argument("--lambda_power", type=float, default=1e5, help="Weight for global power cost")
    parser.add_argument("--lambda_area", type=float, default=1e-5, help="Weight for area balancing cost")
    
    args = parser.parse_args()
    
    print("Loading data")
    gates, nets = load_data(args.gates, args.nets)
    
    print("Building convex QP problem")
    problem, x, nodes = build_problem(gates, nets, args.layers,
                                      lambda_edge=args.lambda_edge,
                                      lambda_power=args.lambda_power,
                                      lambda_area=args.lambda_area,
                                      p_edge_cross=p_edge_cross,
                                      p_intra_net=p_intra_net)
    
    print("Solving QP problem")
    try:
        problem.solve(solver=cp.OSQP, verbose=True, max_iter=10000)
    except Exception as e:
        print("Solver Failed:", e)
        return
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem did not solve to optimality. Status:", problem.status)
        return
    
    print("Retrieving partition results")
    partitions = get_partitioning(x, nodes, args.layers)
    stats = compute_partition_stats(partitions, gates, nets, p_intra_net)
    
    # Build a reverse mapping from node to its assigned layer
    node_to_layer = {}
    for layer, node_list in partitions.items():
        for node in node_list:
            node_to_layer[node] = layer
    
    # Count the number of edge crossings (a crossing is counted for each unique pair of nodes in the same net that are in different layers)
    edge_crossings = count_edge_crossings(nets, node_to_layer)
    
    # Count the number of net crossings
    net_crossings = count_net_crossings(nets, node_to_layer)
    
    # Combine partition stats and the crossing count into one result dictionary.
    result = {
        "partitions": stats,
        "edge_crossings": edge_crossings,
        "net_crossings": net_crossings, 
        "edge_crossing_power": p_edge_cross * net_crossings
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print("Partitioning completed. Results written to", args.output)
    print("Edge crossings count:", edge_crossings)

if __name__ == "__main__":
    main()
