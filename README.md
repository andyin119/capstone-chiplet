# Gate-to-Graph JSON Generator

This tool parses a Verilog netlist and a gate report to generate two JSON files:

- `gate_instances.json`: List of gate instances with area and leakage power.
- `net_connections.json`: List of net connections between gates.

## Requirements
Flattened netlist of design placed in build dir
- `ChipTop.flat.v` 

Report files generated from synthesis placed in build dir
- `power_per_cell.rpt`
- `final_gates.rpt`

## How to Run

To generate the graph JSON files, navigate to the `build` directory and simply run:

```bash
make run
```

# Partition-to-Gate Netlist Generator

This tool parses a text file describing the partition and generates netlists for each layer from a larger design. For N layers, it generates N netlist files:

- `ChipTop_layerN.v`: Netlist with gates only from partition

