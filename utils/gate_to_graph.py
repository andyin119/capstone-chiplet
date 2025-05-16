import json
import re
from collections import defaultdict
from typing import Dict, List, Set


class GateParser:
    def __init__(
        self, netlist_path: str, gate_rpt_path: str, power_rpt_path: str = None
    ):
        """
        Initialize GateParser.

        Initializes a GateParser object with specified netlist and report file paths.

        Parameters
        ----------
        netlist_path : str
            Path to the netlist file.

        gate_rpt_path : str
            Path to the gate report file.

        Returns
        -------
        None
        """
        self.netlist_path = netlist_path
        self.gate_rpt_path = gate_rpt_path
        self.power_rpt_path = power_rpt_path
        self.gate_library: Dict[str, Dict[str, float]] = {}
        self.gate_instances: List[Dict[str, object]] = []
        self.net_connections: Dict[str, List[str]] = defaultdict(list)

    def parse_gate_reports(self):
        """
        Parse gate report files.

        Parses the gate report file to extract gate area and leakage power information.
        The method reads through the report line-by-line, stopping at the total line,
        and stores the relevant data for each gate type in the gate library.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Open the gate report file and process it line-by-line.
        with open(self.gate_rpt_path) as f:
            for line in f:
                # Stop processing if a line starts with 'total' (case-insensitive)
                if line.strip().lower().startswith("total"):
                    break
                # Match lines with gate name, area, and leakage values.
                match = re.match(r"^\s*(\S+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
                if match:
                    gate, area, leakage = match.groups()
                    # Store the area and leakage power for the gate type.
                    self.gate_library[gate] = {
                        "area": float(area),
                    }

        if self.power_rpt_path:
            with open(self.power_rpt_path) as f:
                for line in f:
                    # Match lines with gate name, area, and leakage values.
                    match = re.match(
                        r"^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([A-Za-z0-9_]+)",
                        line,
                    )
                    if match:
                        leakage, internal, switching, total, gate = match.groups()
                        # Store the area and leakage power for the gate type.
                        if gate not in self.gate_library:
                            print(f"Warning: Gate {gate} not found in library.")
                            continue
                        self.gate_library[gate].update(
                            {
                                "leakage_power": float(leakage),
                                "internal_power": float(internal),
                                "switching_power": float(switching),
                                "total_power": float(total),
                            }
                        )

    def expand_wires(self, wire_str: str):
        """
        Expand wire string.

        Expands a signal string with bus notation into a list of individual signals.
        The method handles bus ranges and single signals, returning a clean list
        of signal names.

        Parameters
        ----------
        wire_str : str
            The wire string possibly containing bus notation.

        Returns
        -------
        List[str]
            A list of individual signal names.
        """
        # Remove leading and trailing whitespace
        wire_str = wire_str.strip()

        # Match optional bus and signal name (with optional index), strip leading backslash
        bus_match = re.match(r"\[(\d+):(\d+)\]\s*(\\?[\w]+(?:\[\d+\])?)", wire_str)

        if bus_match:
            msb = int(bus_match.group(1))
            lsb = int(bus_match.group(2))
            base = bus_match.group(3).lstrip("\\")  # remove optional leading backslash

            expanded = []
            for i in range(lsb, msb + 1):
                expanded.append(f"{base}[{i}]")  # e.g., basic[i]
            return expanded
        else:
            # No bus, just clean up and return signal as-is
            return [wire_str.lstrip("\\")]

    def store_wires(self, wire_str: str):
        """
        Store wires from declaration.

        Processes a wire declaration line to extract and store individual signal names.
        The method cleans the declaration and expands any bus notation to ensure
        all signals are recorded.

        Parameters
        ----------
        wire_str : str
            The wire declaration string.

        Returns
        -------
        None
        """
        # strip wire declaration
        wire_str = re.sub(r"^\s*wire\s+", "", wire_str)
        # split by commas
        nets = [net.strip() for net in wire_str.split(",")]

        for net in nets:
            # Expand bus notation into individual signals
            expanded_signals = self.expand_wires(net)
            # Clean up each signal name by removing escape characters and indices
            for signal in expanded_signals:
                # Add the cleaned-up signal to the net_connections dictionary
                if signal not in self.net_connections:
                    self.net_connections[signal] = []

    def store_connections(self, inst: str, ports_str: str):
        """
        Store gate instance connections.

        Stores the connections of a gate instance to its nets based on port string.
        The method extracts nets from the port connections and manages packed wires,
        ensuring all connections are appropriately recorded.

        Parameters
        ----------
        inst : str
            The instance name.

        ports_str : str
            The string containing port connections.

        Returns
        -------
        None
        """
        # Find all nets used in the port connections (pattern: .<port>(net_name)).
        nets_in_inst = re.findall(r"\(([^)]*)\)", ports_str)

        # Warn if no nets were found for this instance.
        if not nets_in_inst:
            print(f"Warning: No nets found in instance {inst}")

        # Check for packed wires in the nets and expand them
        expanded_nets = []
        for net in nets_in_inst:
            if "{" in net and "}" in net:
                # Extract the packed wires inside curly braces
                packed_nets = re.findall(r"\{([^}]+)\}", net)
                for packed_net in packed_nets:
                    # Split the packed nets by commas and add them to the expanded list
                    expanded_nets.extend([n.strip() for n in packed_net.split(",")])
            else:
                # Add the net as-is if it's not packed
                expanded_nets.append(net)

        # Process each net extracted from the port list.
        for net in expanded_nets:
            net = net.replace(" ", "")
            # Clean up the net name by removing escape characters and indices.
            if net.startswith("\\"):
                net = net[1:]

            # If the net is already declared, add the instance to its connections list.
            if net in self.net_connections:
                if inst not in self.net_connections[net]:
                    self.net_connections[net].append(inst)

    def store_instances(self, inst: str):
        """
        Store gate instance.

        Stores a gate instance along with its parameters and connections.
        The method parses the instance declaration, retrieves gate information,
        and records the instance in the gate instances list.

        Parameters
        ----------
        inst : str
            The gate instance declaration string.

        Returns
        -------
        None
        """
        # Match a gate instantiation of the form: <gate_type> <instance_name> ( ... )
        # Skip if the line does not match the expected format.
        header_match = re.match(r"^\s*(\w+)\s+(?:\\)?([^\(\s]+)\s*\(", inst)
        if not header_match:
            print(f"Warning: Unable to parse instance line: {inst}")
            return

        # Extract the gate type and instance name from the match.
        gate_type, inst_name = header_match.groups()
        # Get gate information from the gate library, use -1 if not found.
        gate_info = self.gate_library.get(
            gate_type,
            {
                "area": -1,
                "leakage_power": -1,
                "internal_power": -1,
                "switching_power": -1,
                "total_power": -1,
            },
        )
        # Record the gate instance with its associated parameters.
        self.gate_instances.append(
            {
                "node": inst_name,
                "gate": gate_type,
                "area": gate_info["area"],
                "leakage_power": gate_info.get("leakage_power", -1),
                "internal_power": gate_info.get("internal_power", -1),
                "switching_power": gate_info.get("switching_power", -1),
                "total_power": gate_info.get("total_power", -1),
            }
        )

        # Extract the port list (content between the first '(' and the last ')').
        try:
            ports_str = inst.split("(", 1)[1].rsplit(")", 1)[0]
        except IndexError:
            print(f"Error parsing ports for instance {inst_name} of type {gate_type}")
            return

        # Store the instance in the net_connections dictionary for each net it connects to.
        self.store_connections(inst_name, ports_str)

    def parse_netlist(self):
        """
        Parse verilog netlist.

        Parses the verilog netlist to extract gate instances and net connections from the ChipTop module.
        The method reads the file, identifies the ChipTop module, and processes its content to
        store instances and connections appropriately.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Read the entire netlist file contents.
        with open(self.netlist_path) as f:
            content = f.read()
        # Extract the content inside the ChipTop module.
        module_match = re.search(
            r"module\s+ChipTop.*?;(.*?)endmodule", content, re.DOTALL
        )
        if not module_match:
            print("ChipTop module not found in netlist file.")
            return
        module_content = module_match.group(1)

        # Initialize net_connections with declared wires.
        self.net_connections = defaultdict(list)

        # Split the module content into individual statements by semicolon.
        instances = module_content.split(";")
        for inst in instances:
            inst = inst.replace("\n", " ").strip()

            # Check if the line is a wire declaration or an instance.
            # If it's a wire declaration, store the wires.
            # If it's an instance, store the instance and its connections.
            if inst.startswith("wire"):
                self.store_wires(inst)
            elif inst and "(" in inst:
                self.store_instances(inst)
            else:
                print(f"Warning: Unrecognized line in module content: {inst}")

    def write_gate_instances(self, output_path: str):
        """
        Write gate instances to JSON.

        Writes the list of gate instances to a specified JSON file.
        The method formats the instances into JSON and saves them to the given path.

        Parameters
        ----------
        output_path : str
            The file path to write gate instances JSON.

        Returns
        -------
        None
        """
        with open(output_path, "w") as f:
            json.dump(self.gate_instances, f, indent=2)

    def write_net_connections(self, output_path: str):
        """
        Write net connections to JSON.

        Formats and writes the net connections to a specified JSON file.
        The method prepares the connections data and saves it in a structured JSON format.

        Parameters
        ----------
        output_path : str
            The file path to write net connections JSON.

        Returns
        -------
        None
        """
        formatted = [
            {"net": net, "connections": gates}
            for net, gates in self.net_connections.items()
        ]
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)

    def run(self, gate_output: str, net_output: str):
        """
        Run gate parsing process.

        Executes the parsing of the gate report and netlist, and writes
        the results to JSON files. This method orchestrates the overall parsing
        workflow and provides summary information about the parsed data.

        Parameters
        ----------
        gate_output : str
            The file path for gate instances JSON.

        net_output : str
            The file path for net connections JSON.

        Returns
        -------
        None
        """
        # Parse the gate report to build the gate library.
        self.parse_gate_reports()
        # Parse the netlist to extract gate instances and net connections.
        self.parse_netlist()
        # Print summary information.
        print(f"Parsed {len(self.gate_instances)} gate instances.")
        total_nets = sum(len(conns) for conns in self.net_connections.values())
        print(
            f"Parsed {total_nets} net connections across {len(self.net_connections)} nets."
        )
        # Write the parsed data to JSON output files.
        self.write_gate_instances(gate_output)
        self.write_net_connections(net_output)
        print(f"Wrote {gate_output} and {net_output}")


def main():
    """
    Main entry point.

    Parses command-line arguments and runs the GateParser to generate JSON outputs.
    This method sets up the argument parser and initiates the parsing process
    based on user-specified file paths.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    import argparse

    # Set up the argument parser and define command-line options.
    # example usage: python gate_to_graph.py --netlist ChipTop.flat.v --rpt final_gates.rpt
    parser = argparse.ArgumentParser(description="Parse netlist and report to JSON.")
    parser.add_argument("--netlist", required=True, help="Path to ChipTop.flat.v")
    parser.add_argument("--area_rpt", required=True, help="Path to final_gates.rpt")
    parser.add_argument("--pwr_rpt", required=False, help="Path to power_per_cell.rpt")
    parser.add_argument(
        "--gate_json",
        default="gate_instances.json",
        help="Output JSON file for gate instances",
    )
    parser.add_argument(
        "--net_json",
        default="net_connections.json",
        help="Output JSON file for net connections",
    )

    args = parser.parse_args()

    # Create a GateParser instance with the provided file paths.
    if args.pwr_rpt:
        gp = GateParser(args.netlist, args.area_rpt, args.pwr_rpt)
    else:
        gp = GateParser(args.netlist, args.rpt)
    # Run the parser to generate the JSON outputs.
    gp.run(args.gate_json, args.net_json)


if __name__ == "__main__":
    main()
