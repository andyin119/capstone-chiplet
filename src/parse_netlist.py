#!/usr/bin/env python3
import re
import json
import argparse

def parse_lvs_netlist(lvs_filename):
    """
    Parses the LVS netlist file to extract cell information.
    Assumes each cell instantiation is of the form:
      <cell_type> <full_path_instance_name> ( .port1(net1), .port2(net2), ... );
    Returns a list of tuples:
      (cell_type, full_path_instance_name, [net1, net2, ...])
    """
    cells = []
    # Updated regex:
    #   group(1): cell type (letters, digits, underscore)
    #   group(2): instance name (any non-space characters up to '(')
    #   group(3): connection list inside the parentheses
    cell_inst_pattern = re.compile(
        r'^\s*([A-Za-z0-9_]+)\s+([^\s(]+)\s*\(\s*(.*?)\s*\)\s*;',
        re.DOTALL | re.MULTILINE
    )
    # Pattern to extract nets from connection list: matches .<port>(net)
    net_pattern = re.compile(r'\.\w+\s*\(\s*([^)]+)\s*\)')
    
    with open(lvs_filename, 'r') as f:
        content = f.read()
        for match in cell_inst_pattern.finditer(content):
            cell_type = match.group(1)
            # Ignore fill cells
            if cell_type.startswith("FILL"):
                continue
            full_path = match.group(2)
            if full_path.startswith("ChipTop"):
                continue
            connections_str = match.group(3)
            nets = net_pattern.findall(connections_str)
            nets = [net.strip() for net in nets]
            cells.append((cell_type, full_path, nets))
    return cells

def parse_lib_file(lib_filename):
    """
    Parses the .lib file to build a mapping from library module names to their
    area and cell leakage power.
    Returns a dict: { module_name: {"area": float, "leakage": float } }
    """
    lib_map = {}
    # Module definition: cell ( <module_name> ) {
    module_pattern = re.compile(r'\bcell\s*\(\s*(\S+)\s*\)\s*\{')
    area_pattern = re.compile(r'\barea\s*:\s*([\d\.Ee+-]+)')
    leakage_pattern = re.compile(r'\bcell_leakage_power\s*:\s*([\d\.Ee+-]+)')
    
    current_module = None
    area = None
    leakage = None
    inside_module = False
    
    with open(lib_filename, 'r') as f:
        for line in f:
            m = module_pattern.search(line)
            if m:
                if current_module is not None:
                    lib_map[current_module] = {"area": area, "leakage": leakage}
                current_module = m.group(1).strip("\"")
                area = None
                leakage = None
                inside_module = True
                continue

            if inside_module:
                a_match = area_pattern.search(line)
                if a_match:
                    try:
                        area = float(a_match.group(1))
                    except ValueError:
                        area = None
                l_match = leakage_pattern.search(line)
                if l_match:
                    try:
                        leakage = float(l_match.group(1))
                    except ValueError:
                        leakage = None
                if line.startswith("}"):
                    if current_module is not None:
                        lib_map[current_module] = {"area": area, "leakage": leakage}
                    current_module = None
                    inside_module = False
    return lib_map

def normalize_name(name):
    """
    Normalize a cell name by converting it to lowercase and removing non-alphanumeric characters.
    """
    return ''.join(ch for ch in name.lower() if ch.isalnum())

def lvs_to_lib_module(lvs_cell_type, lib_map, lib_prefix="sky130_fd_sc_hd__"):
    """
    Converts an LVS cell type (e.g. AND3X2, CLKINVX2, LPFLOW_LSBUF_LH_HL_ISOWELL_TAP_2)
    into a candidate library module name.
    
    The function attempts several approaches:
    
    1. Manual mapping: if a manual mapping exists, use it.
    2. Conventional conversion:
       - If the cell type matches pattern like <letters><digits>X<digits> (e.g., AND3X2),
         convert to "and3_2" with proper prefix.
       - If the cell type matches <letters><digits> (e.g., OR2), convert to "or2" with prefix.
    3. Direct mapping: lowercase the LVS cell type and prepend the prefix.
    4. Normalized matching: compare normalized names (removing non-alphanumeric characters)
       with those in the lib_map.
    
    Returns the best-matched library module name.
    """
    manual_mappings = {
        # Add manual mappings here as needed.
        # Example: "SPECIALCELL": "sky130_fd_sc_hd__special_cell",
    }
    
    # Step 1: Check manual mapping (exact match)
    if lvs_cell_type in manual_mappings:
        return manual_mappings[lvs_cell_type]
    
    # Step 2: Conventional conversion patterns.
    # Pattern A: Letters + digits + 'X' + digits (e.g., AND3X2 -> and3_2)
    m = re.match(r'^([A-Za-z_]+)(\d*)X(\d+)$', lvs_cell_type)
    if m:
        candidate = f"{lib_prefix}{m.group(1).lower()}{m.group(2)}_{m.group(3)}"
        if candidate in lib_map:
            return candidate

    # Step 3: Direct mapping.
    candidate = lib_prefix + lvs_cell_type.lower()
    if candidate in lib_map:
        return candidate

    # Step 4: Normalized matching.
    norm_lvs = normalize_name(lvs_cell_type)
    for key in lib_map.keys():
        # Remove prefix if it exists, then normalize.
        norm_key = key.lower()
        if norm_key.startswith(lib_prefix):
            norm_key = norm_key[len(lib_prefix):]
        norm_key = normalize_name(norm_key)
        if norm_lvs == norm_key:
            return key

    # Fallback: return the direct mapping candidate even if not in lib_map.
    return None

def build_hypergraph(cells):
    """
    Build a hypergraph from the list of cells.
    The hypergraph is a dictionary where:
      key = net name (edge)
      value = list of full cell instance names (nodes) connected to that net.
    Nets named VDD or VSS (case-insensitive) are ignored.
    """
    hypergraph = {}
    ignore_nets = {"VDD", "VSS"}
    
    for _, full_path, nets in cells:
        filtered_nets = [net for net in nets if net.upper() not in ignore_nets]
        for net in filtered_nets:
            if net not in hypergraph:
                hypergraph[net] = []
            hypergraph[net].append(full_path)

    return hypergraph

def main():
    parser = argparse.ArgumentParser(
        description="Parse LVS netlist and lib file to produce a cell JSON and hypergraph file."
    )
    parser.add_argument("-lvs", "--lvsfile", required=True, help="Input LVS netlist file (e.g., ChipTop.lvs.v)")
    parser.add_argument("-lib", "--libfile", required=True, help="Input library file (e.g., sky130_fd_sc_hd__tt_025C_1v80.lib)")
    parser.add_argument("-j", "--jsonout", required=True, help="Output JSON file for cell data (e.g., cells.json)")
    parser.add_argument("-hgr", "--hgrout", required=True, help="Output hypergraph file (e.g., ChipTop.hgr)")
    
    args = parser.parse_args()
    
    # Parse LVS netlist.
    print(f"Parsing LVS netlist file: {args.lvsfile}")
    lvs_cells = parse_lvs_netlist(args.lvsfile)
    print(f"Found {len(lvs_cells)} cell instantiations.")
    
    # Parse library file.
    print(f"Parsing library file: {args.libfile}")
    lib_map = parse_lib_file(args.libfile)
    print(f"Found {len(lib_map)} modules in the library file.")

    # Build cell dictionary.
    cells_dict = {}
    for cell_type, full_path, nets in lvs_cells:
        candidate = lvs_to_lib_module(cell_type, lib_map)
        lib_info = lib_map.get(candidate, {"area": None, "leakage": None})
        filtered_nets = [net for net in nets if net.upper() not in {"VDD", "VSS"}]
        cells_dict[full_path] = {
            "cell_type": cell_type,
            "area": lib_info["area"],
            "leakage": lib_info["leakage"],
            "nets": filtered_nets
        }
    
    # Dump the cell dictionary to JSON.
    with open(args.jsonout, "w") as out_f:
        json.dump(cells_dict, out_f, indent=2)
    print(f"Cell dictionary dumped to {args.jsonout}")
    
    # Build hypergraph.
    hypergraph = build_hypergraph(lvs_cells)
    
    # Write hypergraph to file: each line shows net: connected cell instance names.
    with open(args.hgrout, "w") as hg_f:
        hg_f.write(f"p {len(lvs_cells)} {len(hypergraph)}\n")
        for net, cells in hypergraph.items():
            line = f"{net} " + " ".join(cells) + "\n"
            hg_f.write(line)
    print(f"Hypergraph dumped to {args.hgrout}")

if __name__ == "__main__":
    main()
