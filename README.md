# Epidemic Simulation on Complex Networks

This project is part of my Master's Final Project (TFM), focused on simulating the spread of an epidemic over a network and evaluating the impact of different containment strategies. The goal is to understand where and how to apply limited resources to reduce the **attack rate** and effectively counteract the epidemic.

## Project Objective

The main objective is to simulate epidemic propagation in a network and analyze:

- The effectiveness of different containment strategies.
- Where interventions should be applied within the network.
- What parameters are optimal for each strategy to work.

The project explores how various methods perform under resource constraints and different initial conditions (e.g., number of infected nodes).

##  Model Description

- The population is modeled as a complex network (graph), where each node represents an agent.
- Each agent can be in one of several states:
  - `S`: Susceptible
  - `I`: Infected
  - `R`: Recovered
  - `P`: Protected
- At each time step:
  - A limited number of agents are tested (simulating resource constraints).
  - Upon detecting an infected agent, containment strategies are applied.
  - The epidemic progresses according to the contact network.

## Containment Strategies Implemented

- **Random Testing:** Randomly test a small number of agents to detect infections.
- **Protection by Betweenness Centrality:** Protect nodes with the highest betweenness scores to block major transmission paths.
- **Neighbor-Based Intervention:** Apply resources to the most connected neighbors of newly detected infected nodes.
- **High-Degree Node Protection:** (Global): Protect the nodes with the highest degree (most connections) in the    entire network.
- **Random Protection:** A number of agents are randomly selected and protected.

Each strategy is evaluated based on its ability to reduce the total number of infections and control the spread.

## Parameters Explored

- Number of initial infected nodes (seeds).
- Network topology (random, scale-free, etc.).
- Available resources (e.g., number of tests or interventions per time step).
- Strategy-specific parameters (e.g., how many neighbors to protect).

## Technologies Used

- **Python 3**
- **NetworkX** – for network creation and analysis
- **Matplotlib / Seaborn** – for visualization (if applicable)
- **Random / Numpy** – for stochastic simulation


