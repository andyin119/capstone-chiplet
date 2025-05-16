# Running with Chipyard and Genus

This directory supports flows that begin with Cadence Genus, and Chipyard to produce flattened netlists and associated reports.

## Generating Netlists and Reports with Genus

Chipyard’s standard synthesis flow with Cadence Genus is used to produce hierarchical netlists for different designs. To enable parsing and partitioning, you must first flatten the design and generate power reports. While we cannot share the exact custom TCL script, the key Genus commands are:

```text
report_power -by_libcell > power_per_cell.rpt
ungroup -all -flatten -force
write_hdl > ChipTop.flat.v
```

## Setup Instructions

Place the following files into this `data/build` directory **before** running any parsing or partitioning scripts:

- `ChipTop.flat.v` – Flattened top-level Verilog netlist  
- `power_per_cell.rpt` – Power report grouped by library cell  
- `final_gates.rpt` – Area report grouped by library cell  

These files are required inputs for downstream scripts that generate and partition the netlist's hypergraph representation.