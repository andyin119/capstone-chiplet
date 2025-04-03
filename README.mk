# Gate-to-Graph JSON Generator

This tool parses a Verilog netlist and a gate report to generate two JSON files:

- `gate_instances.json`: List of gate instances with area and leakage power.
- `net_connections.json`: List of net connections between gates.

## How to Run

To generate the graph JSON files, navigate to the `build/TinyRocketConfig` directory and simply run:

```bash
make run
