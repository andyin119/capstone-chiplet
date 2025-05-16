# Gate-to-Graph JSON Generator

This tool parses a Verilog netlist and a gate report to generate two JSON files:
- `data/gate_instances.json`: List of gate instances with area and leakage power.
- `data/net_connections.json`: List of net connections between gates.

# Partition-to-Gate Netlist Generator

This tool parses a text file describing the partition and generates netlists for each layer from a larger design. For N layers, it generates N netlist files:
- `ChipTop_layerN.v`: Netlist with gates only from partition