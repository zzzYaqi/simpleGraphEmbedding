from gensim.models import Word2Vec
import random


class deepWalk:
    def __init__(self, graph, walk_length, num_walks, dimensions, window_size, epochs):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = 2
        self.epochs = epochs

    def _random_walk(self, start_node):
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            next_nodes = list(self.graph.neighbors(current))
            if next_nodes:
                walk.append(random.choice(next_nodes))
            else:
                break
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes)
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._random_walk(node))
        return walks

    def train_model(self, walks):
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size,
                         min_count=0, sg=1, workers=self.workers, epochs=self.epochs)
        return model