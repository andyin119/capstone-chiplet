import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
with open("data/subsampled_data/gates_5000.json") as f:
    gate_instances = json.load(f)
with open("data/subsampled_data/nets_5000.json") as f:
    net_connections = json.load(f)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Config
tiers = 3
middle_layer = tiers // 2
gate_names = [g["node"] for g in gate_instances if "node" in g]
gate_index = {name: i for i, name in enumerate(gate_names)}
num_gates = len(gate_names)
area_arr = np.array([g.get("area", 0.0) for g in gate_instances])
power_arr = np.array([g.get("total_power", 0.0) for g in gate_instances])
w1, w2, w3 = 1.0, 1e6, 1e3

x = torch.tensor(np.stack((area_arr, power_arr), axis=1), dtype=torch.float32).to(
    device
)

BITS_PER_NET = 1
TSV_ENERGY_PJ = 0.5
NET_ENERGY_PJ = 0.35

TSV_POWER = TSV_ENERGY_PJ * BITS_PER_NET * 1e-12 / 1e-9
NET_POWER = NET_ENERGY_PJ * BITS_PER_NET * 1e-12 / 1e-9


def compute_cost(partition):
    partition_arr = np.array(partition)
    layer_counts = np.bincount(partition_arr, minlength=tiers)
    layer_areas = np.zeros(tiers)
    layer_power = np.zeros(tiers)
    for t in range(tiers):
        idx = partition_arr == t
        layer_areas[t] = np.sum(area_arr[idx])
        layer_power[t] = np.sum(power_arr[idx])
    avg_area = np.mean(layer_areas)
    raw_area_imbalance = np.sum((layer_areas - avg_area) ** 2)
    area_imbalance = raw_area_imbalance / (np.sum(layer_areas) ** 2 + 1e-8)

    net_crossings = 0
    net_power = 0.0
    for net in net_connections:
        node_ids = [
            gate_index[n]
            for n in net.get("connections", [])
            if isinstance(n, str) and n in gate_index
        ]
        if not node_ids or max(node_ids) >= len(partition_arr):
            continue
        layers = set(partition_arr[node_ids])
        if len(layers) > 1:
            net_crossings += len(layers) - 1
            net_power += TSV_POWER * (len(layers) - 1)
        else:
            net_power += NET_POWER

    global_power = np.sum(power_arr) + net_power
    mid_power = layer_power[middle_layer]
    penalty = 10.0 * max(0, mid_power - global_power / (tiers + 1))

    total_cost = w1 * net_crossings + w2 * global_power + w3 * area_imbalance + penalty

    if np.any(layer_counts == 0):
        total_cost += 1e6

    return total_cost, net_crossings, global_power, area_imbalance, penalty, layer_power


# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_q_learning(epochs=30, lr=1e-3, epsilon_decay=0.95, gamma=0.9):
    model = QNetwork(input_dim=2, output_dim=tiers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epsilon = 1.0

    state_features = x.cpu().numpy()
    partition = np.random.randint(0, tiers, size=num_gates)
    best_partition = partition.copy()
    best_cost, *_ = compute_cost(partition)

    for epoch in range(epochs):
        for i in range(num_gates):
            state = torch.tensor(state_features[i], dtype=torch.float32, device=device)
            q_values = model(state)

            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, tiers - 1)
            else:
                action = torch.argmax(q_values).item()

            old_layer = partition[i]
            partition[i] = action

            new_cost, *_ = compute_cost(partition)
            reward = (best_cost - new_cost) / (abs(best_cost) + 1e-8)
            reward = max(-1.0, min(1.0, reward))

            with torch.no_grad():
                target_q = reward + gamma * torch.max(q_values)

            loss = (q_values[action] - target_q).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if new_cost < best_cost:
                best_cost = new_cost
                best_partition = partition.copy()
            else:
                partition[i] = old_layer  # revert if worse

        print(f"Epoch {epoch+1}: Best Cost = {best_cost:.2f}")
        epsilon *= epsilon_decay

    return best_partition, best_cost


# Run training
final_partition, final_cost = train_q_learning()
# Save final partition result
layers = {i: [] for i in range(tiers)}
for i, l in enumerate(final_partition):
    layers[l].append(gate_names[i])

with open("data/q_learning_partition_result.txt", "w") as f:
    f.write(f"Total Cost: {final_cost:.6f}\n\n")
    for l in range(tiers):
        f.write(f"Layer {l}:\n")
        f.write("  Nodes:\n")
        for g in layers[l]:
            f.write(f"    {g}\n")
        f.write("\n")


def compute_metrics(partition):
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

    # Power parameters
    BITS_PER_NET = 1
    TSV_POWER = 0.5 * BITS_PER_NET * 1e-12 / 1e-9  # = 0.064 W
    NET_POWER = 0.35 * BITS_PER_NET * 1e-12 / 1e-9  # = 0.0448 W

    net_power = 0.0
    net_crossings = 0
    edge_crossings = 0

    for net in net_connections:
        if not isinstance(net, dict) or "connections" not in net:
            continue
        node_ids = [
            gate_index[n]
            for n in net["connections"]
            if isinstance(n, str) and n in gate_index
        ]
        if not node_ids or max(node_ids) >= len(partition_arr):
            continue

        layers = set(partition_arr[node_ids])
        if len(layers) > 1:
            net_crossings += len(layers) - 1
            net_power += TSV_POWER * (len(layers) - 1)
        else:
            net_power += NET_POWER

        # Edge crossings (pairwise)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if partition_arr[node_ids[i]] != partition_arr[node_ids[j]]:
                    edge_crossings += 1

    base_power = np.sum(power_arr)
    global_power = base_power + net_power
    mid_power = layer_power[middle_layer]
    penalty = 10.0 * max(0, mid_power - global_power / (tiers + 1))

    return {
        "Final Cost": w1 * net_crossings
        + w2 * global_power
        + w3 * area_imbalance
        + penalty,
        "Net Crossings": net_crossings,
        "Edge Crossings": edge_crossings,
        "Global Power (W)": global_power,
        "Area Imbalance (normalized)": area_imbalance,
        "Average Area": avg_area,
        "Area per Layer": layer_areas.tolist(),
        "Power per Layer": layer_power.tolist(),
        "Constraint Penalty": penalty,
    }


# Print metrics
print("\n🔍 Final Metrics:")
metrics = compute_metrics(final_partition)
for k, v in metrics.items():
    if k == "Power per Layer":
        for i, p in enumerate(v):
            print(f"  Layer {i} Power (W): {p:.6f}")
    else:
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
