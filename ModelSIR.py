import pandas as pd
from mesa import Agent, Model
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


class Person(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = "S"  # Susceptible por defecto


class ModelSIR(Model):
    def __init__(self, N, infection_prob, recovery_time, m, model="BA", lam=-1, error_prob=0.0):
        super().__init__() 
        self.num_agents = N
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.lam = lam
        self.m = m

        if model == "BA":
            self.network = nx.barabasi_albert_graph(N, m)
        else:
            self.network = nx.erdos_renyi_graph(N, m / N)

        self.set_Lists()

    def set_Lists(self):
        self.susceptibles = []
        self.infected = []
        self.recovered = []

        initial_infected = set(np.random.choice(range(self.num_agents), size=10, replace=False))

        for i in range(self.num_agents):
            agent = Person(i, self)
            if i in initial_infected:
                agent.state = "I"
                self.infected.append(agent)
            else:
                self.susceptibles.append(agent)

    def step(self):
        time = 0
        beta = self.infection_prob if self.lam < 0 else self.lam
        mu = 1 / self.recovery_time
        index = 0

        dataFrame = pd.DataFrame(columns=['time', 'S', 'I', 'R', 'attack_rate'])

        while len(self.infected) > 0:
            new_infected = []
            new_recovered = []

            infected_ids = {agent.unique_id for agent in self.infected}
            susceptible_copy = self.susceptibles.copy()

            # Contagio
            for s in self.susceptibles:
                neighbors = list(self.network.neighbors(s.unique_id))
                infected_neighbors = len(infected_ids & set(neighbors))
                if infected_neighbors > 0:
                    p_infected = 1 - ((1 - beta) ** infected_neighbors)
                    if random.random() < p_infected:
                        s.state = "I"
                        new_infected.append(s)
                        susceptible_copy.remove(s)

            # Recuperación
            for i in self.infected:
                if random.random() < mu:
                    i.state = "R"
                    new_recovered.append(i)

            # Actualización de estados
            self.susceptibles = susceptible_copy
            self.infected = [a for a in self.infected if a not in new_recovered] + new_infected
            self.recovered += new_recovered

            attack_rate = float(len(self.recovered)) / float(self.num_agents)
        
            dataFrame.loc[index] = [time, len(self.susceptibles), len(self.infected), len(self.recovered), attack_rate]

            time += 1
            index += 1
        return dataFrame
