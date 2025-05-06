import os
import re

def get_nodes_per_layer(filepath):
    layer_nodes_list = []
    current_layer = None
    collecting = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()

            # Start of a new layer's node section
            match = re.match(r"Layer (\d+) Nodes:", stripped)
            if match:
                current_layer = int(match.group(1))
                # Ensure list is long enough
                while len(layer_nodes_list) <= current_layer:
                    layer_nodes_list.append({})
                collecting = True
                continue

            # Stop collecting at the start of a new section
            if collecting and stripped.startswith("Layer") and "Power" in stripped:
                collecting = False
                continue

            # Add nodes as keys with None as default value
            if collecting and stripped:
                layer_nodes_list[current_layer][stripped] = None

    return layer_nodes_list

def split_netlist_by_layer(netlist_path, partition_path, output_dir):
    with open(netlist_path, 'r') as f:
        content = f.read()

    sections = content.split(';')

    # Keep module declaration and all 'wire' sections
    print(f"partition_path: {partition_path}")
    # Parse node layers
    nodes_per_layer = get_nodes_per_layer(partition_path)
    num_layers = len(nodes_per_layer)


    for layer_num in range(num_layers):
        layer_nodes = set(nodes_per_layer[layer_num])
        layer_file = os.path.join(output_dir, f'ChipTop_layer{layer_num}.v')

        with open(layer_file, 'w') as out_f:
            # Write preserved sections
            for s in sections:
                if s.strip().startswith('endmodule'):
                    out_f.write(s)
                elif 'module' in s or s.strip().startswith('wire') or s.strip().startswith('input') or s.strip().startswith('output') or s.strip().startswith('inout'):
                    out_f.write(s + ';')
                else:
                    header_match = re.match(r"^\s*\w+\s+(?:\\)?([^\(\s]+)\s*\(", s.strip())
                    if header_match:
                        inst_name = header_match.group(1)
                        if inst_name in layer_nodes:
                            out_f.write(s + ';')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--netlist', type=str, required=True, help='Path to ChipTop.flat.v')
    parser.add_argument('--partition', type=str, required=True, help='Path to partition text file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to write layer-separated netlists')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split_netlist_by_layer(args.netlist, args.partition, args.output_dir)