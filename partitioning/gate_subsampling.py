import json
import random
import argparse
from collections import deque

# Subsampling a hypergraph from the gates and nets information

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def build_node_to_nets_map(nets):
    mapping = {}
    for entry in nets:
        net_name = entry.get('net')
        connections = entry.get('connections', []) or []
        for node in connections:
            mapping.setdefault(node, []).append(net_name)
    return mapping

def build_net_lookup(nets):
    return {entry.get('net'): entry for entry in nets}

def build_gate_lookup(gates):
    return {entry.get('node'): entry for entry in gates}

def subsample(gates, nets, target_count, expansion_ratio, seed=None):
    # Lookups
    gate_lookup = build_gate_lookup(gates)
    node_to_nets = build_node_to_nets_map(nets)
    net_lookup = build_net_lookup(nets)

    all_nodes = list(gate_lookup.keys())
    total_nodes = len(all_nodes)
    if target_count > total_nodes:
        raise ValueError(f"Requested {target_count} nodes, but only {total_nodes} available.")

    # Seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Degree-based stratification into quartile bins
    degree_map = {node: len(node_to_nets.get(node, [])) for node in all_nodes}
    sorted_degrees = sorted(degree_map.values())
    q1 = sorted_degrees[int(total_nodes * 1/4)]
    q2 = sorted_degrees[int(total_nodes * 2/4)]
    q3 = sorted_degrees[int(total_nodes * 3/4)]

    bins = {0: [], 1: [], 2: [], 3: []}
    for node, deg in degree_map.items():
        if deg <= q1:
            bins[0].append(node)
        elif deg <= q2:
            bins[1].append(node)
        elif deg <= q3:
            bins[2].append(node)
        else:
            bins[3].append(node)

    desired, fractional = {}, {}
    for b, nodes in bins.items():
        proportion = len(nodes) / total_nodes
        exact = proportion * target_count
        floor_count = int(exact)
        desired[b] = min(floor_count, len(nodes))
        fractional[b] = exact - floor_count

    allocated = sum(desired.values())
    remaining = target_count - allocated
    for b in sorted(fractional, key=lambda x: fractional[x], reverse=True):
        if remaining <= 0:
            break
        if desired[b] < len(bins[b]):
            desired[b] += 1
            remaining -= 1

    initial = []
    for b, nodes in bins.items():
        k = desired[b]
        if k > 0:
            initial.extend(random.sample(nodes, k))

    # Fill or trim if needed
    if len(initial) < target_count:
        leftover = set(all_nodes) - set(initial)
        initial.extend(random.sample(list(leftover), target_count - len(initial)))
    elif len(initial) > target_count:
        initial = random.sample(initial, target_count)

    included_nodes = set(initial)
    included_nets = set()
    processed_nets = set()
    queue = deque(initial)

    # BFS with partial expansion
    while queue:
        node = queue.popleft()
        for net in node_to_nets.get(node, []):
            if net in processed_nets:
                continue
            processed_nets.add(net)
            included_nets.add(net)
            connections = net_lookup[net].get('connections', []) or []
            # Filter out already included
            new_nodes = [n for n in connections if n not in included_nodes]
            # Determine how many to add
            if expansion_ratio < 1.0 and new_nodes:
                count = max(1, int(len(new_nodes) * expansion_ratio))
                selected = random.sample(new_nodes, min(count, len(new_nodes)))
            else:
                selected = new_nodes
            # Add selected nodes
            for other in selected:
                included_nodes.add(other)

    subsampled_gates = [gate_lookup[n] for n in included_nodes if n in gate_lookup]
    subsampled_nets = [net_lookup[n] for n in included_nets if n in net_lookup]
    return subsampled_gates, subsampled_nets

def main():
    parser = argparse.ArgumentParser(
        description="Subsample a circuit hypergraph by degree-stratified initial nodes plus partial net expansion."
    )
    parser.add_argument('--gates', required=True, help='Path to gates.json')
    parser.add_argument('--nets', required=True, help='Path to nets.json')
    parser.add_argument('--nodes', type=int, required=True, help='Number of initial nodes to sample')
    parser.add_argument('--expansion_ratio', type=float, default=0.1,
                        help='Fraction of new nodes from each net to include (0-1)')
    parser.add_argument('--output_gates', help='Output path for subsampled gates')
    parser.add_argument('--output_nets', help='Output path for subsampled nets')
    parser.add_argument('--seed', type=int, default=27463, help='Random seed for reproducibility')
    args = parser.parse_args()

    gates = load_json(args.gates)
    nets = load_json(args.nets)

    subsampled_gates, subsampled_nets = subsample(
        gates, nets, args.nodes * 0.2, expansion_ratio=args.expansion_ratio, seed=args.seed
    )

    print(f"Total nodes included: {len(subsampled_gates)}")
    print(f"Total nets included: {len(subsampled_nets)}")

    save_json(subsampled_gates, f"gates_{args.nodes}.json")
    save_json(subsampled_nets, f"nets_{args.nodes}.json")
    print(f"Wrote {len(subsampled_gates)} gates to gates_{args.nodes}.json")
    print(f"Wrote {len(subsampled_nets)} nets to nets_{args.nodes}.json")

if __name__ == '__main__':
    main()
