# Gate-to-Graph JSON Generator

This tool parses a Verilog netlist and a gate report to generate two JSON files:
- `data/gate_instances.json`: List of gate instances with area and leakage power.
- `data/net_connections.json`: List of net connections between gates.

## Requirements
Flattened netlist of design placed in `data/build` dir
- `ChipTop.flat.v` 

Report files generated from synthesis placed in `data/build` dir
- `power_per_cell.rpt`
- `final_gates.rpt`

## How to Run
To generate the graph JSON files, run the following command from the root of this repo:

```bash
make graph
```
# Partitioning Algorithms



# Partition-to-Gate Netlist Generator

This tool parses a text file describing the partition and generates netlists for each layer from a larger design. For N layers, it generates N netlist files:
- `ChipTop_layerN.v`: Netlist with gates only from partition

## Requirements
Flattened netlist of design placed in `data/build` dir
- `ChipTop.flat.v` 

Partitioned results using all nodes from one of the partitioning algorithms in the form of a text file. The file should be in the `data` dir
- `file_name.txt`

## How to Run
To generate the layer by layer netlists, run the following command from the root of this repo:

```bash
make netlists PTN_FILE=data/file_name.txt
```