import networkx as nx
import csv
from networkx.readwrite import json_graph


class gameGraph:
    def __init__(self):
        self.game_node = []
        self.game_group = []
        self.user_node = []
        self.user_like = []
        self.graph = nx.Graph()  # graph of game
        self.read_node()
        self.generate_graph()

    def get_graph(self) -> nx.Graph:
        return self.graph

    def return_data(self):
        return json_graph.node_link_data(self.graph)

    def read_node(self):
        row_id = 0
        with open('C:/Users/20213/PycharmProjects/graph_embedding/data/gameNode.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.game_node.append(row[0])
                self.game_group.append(row[1])
                row_id += 1
        with open('C:/Users/20213/PycharmProjects/graph_embedding/data/userNode.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.user_node.append(row[0])
                self.user_like.append(row[1:])

    def generate_graph(self):
        node_id = 0
        for _ in self.game_node:
            self.graph.add_node(node_id, name=self.game_node[node_id], group = self.game_group[node_id])
            node_id += 1

        record = {}
        for user in self.user_like:
            for i in range(len(user)):
                for j in range(i+1,len(user)):
                    if (int(user[i]), int(user[j])) in record:
                        record[(int(user[i]), int(user[j]))] += 1
                    else:
                        record[(int(user[i]), int(user[j]))] = 1

        for key, value in record.items():
            if value >= 3:
                self.graph.add_edge(int(key[0]), int(key[1]))

    def get_node_group(self):
        return self.game_group

    def get_node_name(self):
        return self.game_node