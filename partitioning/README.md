## Chiplet Partitioning Algorithms
This folder contains varies chiplet partitioning algorithms using data generated from the parser. The paritioning algorithms takes in gates and nets information which can be treated as a hypergraph. The partitioning algorithms solves a k-way hypergraph partitioning problem. 

There are the 4 supported algorithms
* Simulated Annealing
* Quadratic Programming
* Q-Learning
* Reinforcement Learning

## Running the algorithms

### Simulated Annealing
Sample Command
```
python3 simulated_annealing.py ../data/net_connections.json ../data/gate_instances.json -o sa_result.txt --iterations 10000
```
Other flags:
* `-l`: specify the number of layers we want to partition into, default is 3
* `-t_i`: specify the initial temperature of simulated annealing, default is 500
* `-t_f`: specify the final temperature of simulated annealing, default is 0.1
* (optional) `--plot_cost`: whether to plot the cost trend of simulated annealing at the end

Key Parameters:
There are important parameters and weights that can be adjusted to observe different behaviors of the cost function. At the top of the simulated_annealing.py script, the weights of the cost function can be adjusted. There is a set of default weights currently set. It is recommended to normalize each term's value (i.e. weights * cost_value should be similar in magnitude) before doing any weight tuning

```
# Set weights for the cost functions.
# Adjustable for different perference
weights = {
    'cross': 10,  # weight for net crossings cost
    'cross_count': 1, # weight for cost of the number of crossings (penalty)
    'area': 1e-6,    # weight for area balance cost
    'power': 1e5,    # weight for power-related cost
    'mid_pow_penalty': 1e5    # penalty term for reduce power in the middle layer
}
```

### Quadratic Programming
Quadratic Programming implementation is based on solving a quadratic equation with constraints using CVXPY library and open source OSQP solver (other solvers can also be used through CVXPY library).

Sample Command:
```
python3 quadratic_programming.py --gates ../data/gate_instances.json --nets ../data/net_connections.json --output qp_result.json
```
Other flags:
* `--layers`: specify the number of partitions(layers) to partition into, default is 3
* `--lambda_edge`: specify the weight for the edge crossing term in the cost function, default is 100
* `--lambda_power`: specify the weight for the power term in the cost function, default is 1e5
* `--lambda_area`: specify the weight for the area balancing term in the cost function, default is 1e-5


Key Parameters:
`lambda_edge`, `lambda_power` and `lambda_area` are the main parameters for the cost function that can be adjusted to observe for performance.


**Important:** Quadratic Programming implementation tend to take a long time to complete. It might take days to run on the full problem, it is recommended to test run on smaller datasets located in `data/subsampled_data`

### Q-Learning
Description:
This method applies reinforcement learning using Q-learning and a neural network (2-layer MLP) to learn gate-to-layer assignments that minimize a custom cost function.

Sample Command
```
python3 q_learning.py
```
Configurable Parameters (in script):
	•	epochs: number of training epochs (default: 30)
	•	lr: learning rate (default: 1e-3)
	•	gamma: discount factor (default: 0.9)
	•	epsilon_decay: decay rate for exploration probability (default: 0.95)****
 Output:
	•	Best gate partitioning saved to data/q_learning_partition_result.txt
	•	Final performance metrics printed at the end of training.

Highlights:
	•	Supports GPU acceleration (MPS or CUDA).
	•	Includes per-layer area/power tracking and TSV/net power modeling.
	•	Enforces a power constraint on the middle layer with a penalty term.

### Reinforcement Learning
Sample Command
```
python3 reinforcement_learning.py
```
