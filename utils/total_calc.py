import json

# Replace with your actual file path
json_file_path = '../data/gate_instances.json'

# Load the JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Initialize global metrics
count = 0
total_area = 0.0
total_power = 0.0
leakage_power = 0.0
internal_power = 0.0
swithing_power = 0.0

# Accumulate area and total_power
for item in data:
    count += 1
    total_area += item.get("area", 0.0)
    leakage_power += item.get("leakage_power", 0.0)
    internal_power += item.get("internal_power", 0.0)
    swithing_power += item.get("switching_power", 0.0)
    total_power += item.get("total_power", 0.0)

# Print the results
print(f"Total Count: {count}")
print(f"Global Area: {total_area/5486200000}")
print(f"Global Total Power: {total_power}")
print(f"Global Leakage Power: {leakage_power}")
print(f"Global Internal Power: {internal_power}")
print(f"Global Switching Power: {swithing_power}")