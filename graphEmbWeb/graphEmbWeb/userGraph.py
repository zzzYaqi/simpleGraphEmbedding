import networkx as nx
import csv
from networkx.readwrite import json_graph


class userGraph:
    def __init__(self):
        self.game_node = []
        self.game_group = []
        self.user_node = []
        self.user_like = []
        self.user_group = []
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
                if len(row[1:]) < 9:
                    self.user_group.append(0)
                elif 9 <= len(row[1:]) < 18:
                    self.user_group.append(1)
                else:
                    self.user_group.append(2)

    def generate_graph(self):
        node_id = 0
        for _ in self.user_node:
            self.graph.add_node(node_id, name=self.user_node[node_id], group=self.user_group[node_id])
            node_id += 1

        records = []
        for i in range(len(self.user_like)):
            for j in range(i + 1, len(self.user_like)):
                set1 = set(self.user_like[i])
                set2 = set(self.user_like[j])
                common = set1.intersection(set2)
                if len(common) > 3:
                    records.append([i, j])

        for re in records:
            self.graph.add_edge(re[0], re[1])

    def get_node_group(self):
        return self.user_group

    def get_node_name(self):
        return self.user_node
