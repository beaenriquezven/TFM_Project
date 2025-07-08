import pandas as pd
from mesa import Agent, Model
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from enum import Enum


class ProtectionStrategy(Enum):
    NO_STRATEGY = "no_strategy"
    RANDOM = "random"
    HIGH_DEGREE_NEIGHBOR = "high_degree_neighbor"
    HIGH_DEGREE_NETWORK = "high_degree_network"
    HIGH_BETWEENNESS = "high_betweenness"


class ProtectedPerson(Agent):
    def __init__(self, unique_id, model):
        Agent.__init__(self, unique_id, model)
        self.state = "S" 


class PrModelSIR(Model):
    def __init__(self, N, infection_prob, recovery_time, m, resources, seed_nodes=[],model="BA", protection_strategy=ProtectionStrategy.NO_STRATEGY):
        super().__init__()
        self.num_agents = N
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.m = m
        self.strategy = protection_strategy
        self.resources = resources
        self.seed_nodes = seed_nodes

        # Crear red
        self.network = nx.barabasi_albert_graph(N, m) if model == "BA" else nx.erdos_renyi_graph(N, m / N)

        self.set_Lists()  

    def set_Lists(self):
        self.susceptibles = []
        self.infected = []
        self.recovered = []
        self.protected = []

        for i in range(self.num_agents):
            agent = ProtectedPerson(i, self)
            if i in self.seed_nodes:
                agent.state = "I"
                self.infected.append(agent)
            else:
                self.susceptibles.append(agent)

    def step(self):
        time = 0
        beta = self.infection_prob 
        mu = 1 / self.recovery_time
        index = 0

        df = pd.DataFrame(columns=['time', 'S', 'I', 'R', 'P', 'attack_rate'])
        count_detected = 0
        while len(self.infected) > 0:
            new_infected = []
            new_recovered = []

            infected_ids = {a.unique_id for a in self.infected}
            susceptibles_copy = self.susceptibles.copy()

            attack_rate = float(len(self.recovered)) / self.num_agents
            df.loc[index] = [time, len(self.susceptibles), len(self.infected), len(self.recovered), len(self.protected), attack_rate]

            # Propagación de infección
            for s in self.susceptibles:
                neighbors = list(self.network.neighbors(s.unique_id))
                infected_neighbors = len(infected_ids & set(neighbors))
                if infected_neighbors > 0:
                    p_infected = 1 - ((1 - beta) ** infected_neighbors)
                    if random.random() < p_infected:
                        s.state = "I"
                        new_infected.append(s)
                        susceptibles_copy.remove(s)

            self.susceptibles = susceptibles_copy

            # Recuperación
            for i in self.infected:
                if random.random() < mu:
                    if self.strategy != ProtectionStrategy.NO_STRATEGY:
                        if count_detected == 0:
                            self.apply_strategy_on_step(i)
                        count_detected += 1
                    i.state = "R"
                    new_recovered.append(i)

            self.infected = [a for a in self.infected if a not in new_recovered] + new_infected
            self.recovered += new_recovered

            
            time += 1
            index += 1

        return df
    
    def select_random_infected(self, time, index, new_recovered, count_detected):
        combined = self.susceptibles + self.infected
        test_candidates = random.sample(combined, min(100, len(combined)))

        for a in test_candidates:
            if a.state == "I" and count_detected == 0:
                count_detected += 1
                self.apply_strategy(a)
                break  
        return count_detected

    def apply_strategy_on_step(self, agent_id):
        if self.strategy == ProtectionStrategy.RANDOM:
            self.apply_random_protection()
        elif self.strategy == ProtectionStrategy.HIGH_DEGREE_NETWORK:
            self.apply_high_degree_protection()
        elif self.strategy == ProtectionStrategy.HIGH_BETWEENNESS:
            self.apply_betweenness_protection()
        elif self.strategy == ProtectionStrategy.HIGH_DEGREE_NEIGHBOR:
            self.protect_most_connected_neighbors_extended(agent_id)

    def get_agent_id(self, agent_id):
        for a in self.susceptibles + self.infected + self.recovered:
            if a.unique_id == agent_id:
                return a
        return None

    def apply_random_protection(self):
        candidates = self.susceptibles + self.infected
        random.shuffle(candidates)
        resource = 0
        for a in candidates:
            if resource >= self.resources:
                break
            if a.state == "S":
                self.susceptibles.remove(a)
                a.state = "P"
                self.protected.append(a)
            if a.state == "I":
                self.infected.remove(a)
                a.state = "R"
                self.recovered.append(a)
            resource += 1
           

    def apply_high_degree_protection(self):
        nodes = sorted(self.network.degree(), key=lambda x: x[1], reverse=True)
        resource = 0
        
        for node_id, _ in nodes:
            agent = self.get_agent_id(node_id)
            if agent is None:
                continue
            if agent.state == "R":
                continue
            if resource >= self.resources:
                break
            if agent.state == "S":
                self.susceptibles.remove(agent)
                agent.state = "P"
                self.protected.append(agent)
            if agent.state == "I":
                self.infected.remove(agent)
                agent.state = "R"
                self.recovered.append(agent)
            resource += 1

    def apply_betweenness_protection(self):
        self.betweenness = nx.betweenness_centrality(self.network, k=50)
        top_nodes = sorted(self.betweenness.items(), key=lambda x: x[1], reverse=True)
        resource = 0
        for node_id, _ in top_nodes:
            agent = self.get_agent_id(node_id)
            if agent is None:
                continue
            if agent.state == "R":
                continue
            if resource >= self.resources:
                break
            if agent.state == "S":
                self.susceptibles.remove(agent)
                agent.state = "P"
                self.protected.append(agent)
            if agent.state == "I":
                self.infected.remove(agent)
                agent.state = "R"
                self.recovered.append(agent)
            resource += 1
 
    def protect_most_connected_neighbors_extended(self, agent):
        if self.resources <= 0:
            return

        # Primeros vecinos
        first_neighbors = list(self.network.neighbors(agent.unique_id))

        # Segundos vecinos (excluyendo al agente mismo y sus primeros vecinos)
        second_neighbors = set()
        for neighbor in first_neighbors:
            for second in self.network.neighbors(neighbor):
                if second != agent.unique_id and second not in first_neighbors:
                    second_neighbors.add(second)

        # Todos los candidatos a proteger: primeros + segundos vecinos
        all_candidates = set(first_neighbors).union(second_neighbors)

        # Filtrar candidatos válidos: solo S o I
        valid_candidates = [
            n for n in all_candidates
            if self.get_agent_id(n) and self.get_agent_id(n).state in ["S", "I"]
        ]

        if not valid_candidates:
            return

        # Ordenar por mayor grado de conexión
        most_connected = sorted(valid_candidates, key=lambda v: self.network.degree(v), reverse=True)

        # Aplicar protección
        for node_id in most_connected:
            if self.resources <= 0:
                break

            neighbor_agent = self.get_agent_id(node_id)
            if neighbor_agent.state == "S" and neighbor_agent in self.susceptibles:
                self.susceptibles.remove(neighbor_agent)
                neighbor_agent.state = "P"
                self.protected.append(neighbor_agent)
                self.resources -= 1
            elif neighbor_agent.state == "I" and neighbor_agent in self.infected:
                self.infected.remove(neighbor_agent)
                neighbor_agent.state = "R"
                self.recovered.append(neighbor_agent)
                self.resources -= 1

    

    def protect_most_connected_neighbor(self, agent):
        if  self.resources <= 0:
            return 
        neighbors = list(self.network.neighbors(agent.unique_id))
      
        valid_neighbors = [
            n for n in neighbors
            if self.get_agent_id(n) and self.get_agent_id(n).state in ["S", "I"]
        ]

        if not valid_neighbors:
            return

        most_connected = sorted(valid_neighbors,key=lambda v: self.network.degree(v), reverse=True)
        for node_id in most_connected:
            neighbor_agent = self.get_agent_id(node_id)
            if  self.resources > 0:
                if neighbor_agent.state == "S" and neighbor_agent in self.susceptibles:
                    self.susceptibles.remove(neighbor_agent)
                    neighbor_agent.state = "P"
                    self.protected.append(neighbor_agent)
                elif neighbor_agent.state == "I" and neighbor_agent in self.infected:
                    self.infected.remove(neighbor_agent)
                    neighbor_agent.state = "R"
                    self.recovered.append(neighbor_agent)
            self.resources -= 1  

