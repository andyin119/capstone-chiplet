# Path Variables
NETLIST := data/build/ChipTop.flat.v
AREA_RPT := data/build/final_gates.rpt
PWR_RPT := data/build/power_per_cell.rpt


# ============================
# Section 1: PARSING
# ============================

# Rule to generate hypergraph JSON from Verilog netlist
graph:
	# Parses the input Verilog netlist into a hypergraph representation
	python3 parser/gate_to_graph.py \
		--netlist $(NETLIST) \
		--area_rpt $(AREA_RPT) \
		--pwr_rpt $(PWR_RPT)


# ============================
# Section 2: PARTITIONING
# ============================

# Rule to run the specified partitioning algorithm
partition:
	# Partitions the hypergraph JSON into chiplet layers


# ============================
# Section 3: CREATING NETLISTS
# ============================

# Rule to generate per-layer netlists from partition output
netlists:
	# Creates layer-specific Verilog netlists based on the partition
	python3 parser/partition_to_gate.py \
		--netlist $(NETLIST) \
		--partition $(PTN_FILE)