import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# Auto-select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU via MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")


# Load JSON files
with open("data/gate_instances.json") as f:
    gate_instances = json.load(f)
with open("data/net_connections.json") as f:
    net_connections = json.load(f)

# Parse data
gate_names = [g["node"] for g in gate_instances]
gate_index = {name: i for i, name in enumerate(gate_names)}
area_arr = np.array([g["area"] for g in gate_instances])
power_arr = np.array([g["total_power"] for g in gate_instances])
num_gates = len(gate_names)

# Hyperparameters
tiers = 3
middle_layer = 1

BITS_PER_NET = 1
TSV_POWER = 0.5 * BITS_PER_NET * 1e-12 / 1e-9
NET_POWER = 0.35 * BITS_PER_NET * 1e-12 / 1e-9

# Normalize input
features = np.stack((area_arr, power_arr), axis=1)
features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
x = torch.tensor(features, dtype=torch.float32).to(device)


# RL Model
class RLModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=tiers):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Cost function
def compute_cost(partition, area_arr, power_arr, net_connections, gate_index):
    partition_arr = np.array(partition)
    layer_areas = np.zeros(tiers)
    layer_power = np.zeros(tiers)

    for t in range(tiers):
        idx = partition_arr == t
        layer_areas[t] = np.sum(area_arr[idx])
        layer_power[t] = np.sum(power_arr[idx])

    avg_area = np.mean(layer_areas)
    area_imbalance = np.sum((layer_areas - avg_area) ** 2) / (
        np.sum(layer_areas) ** 2 + 1e-8
    )

    # Power model parameters
    BITS_PER_NET = 1
    TSV_POWER = 0.5 * BITS_PER_NET * 1e-12 / 1e-9  # = 0.064 W
    NET_POWER = 0.35 * BITS_PER_NET * 1e-12 / 1e-9  # = 0.0448 W

    net_power = 0.0
    net_crossings = 0
    edge_crossings = 0  # still tracked if needed for evaluation

    for net in net_connections:
        if not isinstance(net, dict) or "connections" not in net:
            continue

        node_ids = [gate_index[n] for n in net["connections"] if n in gate_index]
        if not node_ids:
            continue

        layers = set(partition_arr[node_ids])
        if len(layers) > 1:
            net_crossings += len(layers) - 1
            net_power += TSV_POWER * (len(layers) - 1)
        else:
            net_power += NET_POWER

    base_power = np.sum(power_arr)
    global_power = base_power + net_power
    penalty = 50.0 * max(0, layer_power[middle_layer] - global_power / (tiers + 1))
    w1, w2, w3 = 10, 1e6, 100
    total_cost = w1 * net_crossings + w2 * global_power + w3 * area_imbalance + penalty

    return (
        total_cost,
        net_crossings,
        edge_crossings,
        global_power,
        area_imbalance,
        penalty,
        layer_power,
        layer_areas,
    )


# Training loop
def train_rl(x, area_arr, power_arr, net_connections, gate_index, epochs=100, lr=1e-3):
    model = RLModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_partition = np.random.randint(0, tiers, size=num_gates)
    best_cost, *_ = compute_cost(
        best_partition, area_arr, power_arr, net_connections, gate_index
    )
    for epoch in range(epochs):
        model.train()
        logits = model(x)
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = torch.softmax(logits, dim=1).clamp(min=1e-8)
        actions = torch.multinomial(probs, 1).squeeze()
        partition = actions.cpu().numpy()
        cost, *_ = compute_cost(
            partition, area_arr, power_arr, net_connections, gate_index
        )
        reward = best_cost - cost
        log_probs = torch.log(probs[range(num_gates), actions])
        loss = -log_probs.mean() * reward
        opt.zero_grad()
        loss.backward()
        opt.step()
        if cost < best_cost:
            best_cost = cost
            best_partition = partition.copy()
        print(f"Epoch {epoch+1}: Best Cost = {best_cost:.2f}")
    return best_partition, best_cost, model


# Run training
final_partition, final_cost, model = train_rl(
    x, area_arr, power_arr, net_connections, gate_index
)

# Final metrics
cost, net_x, edge_x, power, imbalance, penalty, l_power, l_areas = compute_cost(
    final_partition, area_arr, power_arr, net_connections, gate_index
)

print("\n🔍 Final Metrics:")
print(f"  Final Cost: {cost:.6f}")
print(f"  Edge Crossings: {edge_x:.0f}")
print(f"  Net Crossings: {net_x:.0f}")
print(f"  Global Power (W): {power:.6f}")
print(f"  Area Imbalance (norm): {imbalance:.6f}")
print(f"  Penalty: {penalty:.6f}")
for i in range(tiers):
    print(f"  Layer {i} Power (W): {l_power[i]:.6f}")
    print(f"  Layer {i} Area: {l_areas[i]:.2f}")

# Save final partition result to txt
layers = {i: [] for i in range(tiers)}
for i, l in enumerate(final_partition):
    layers[l].append(gate_names[i])  # Assuming gate_names exists

with open("data/rl_partition_result.txt", "w") as f:
    f.write(f"Total Cost: {cost:.6f}\n")
    f.write(f"Edge Crossings: {edge_x}\n")
    f.write(f"Net Crossings: {net_x}\n")
    f.write(f"Global Power (W): {power:.6f}\n")
    f.write(f"Area Imbalance (norm): {imbalance:.6f}\n")
    f.write(f"Penalty: {penalty:.6f}\n\n")
    for i in range(tiers):
        f.write(f"Layer {i} Power (W): {l_power[i]:.6f}\n")
        f.write(f"Layer {i} Area: {l_areas[i]:.2f}\n")
        f.write(f"Layer {i} Nodes:\n")
        for g in layers[i]:
            f.write(f"  {g}\n")
        f.write("\n")
