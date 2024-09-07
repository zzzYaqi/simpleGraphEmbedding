import random
from gensim.models import Word2Vec


class node2Vec:
    def __init__(self, graph, walk_length, num_walks, dimensions, window_size, epochs, p, q):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = 4
        self.epochs = epochs
        self.p = p
        self.q = q

    def get_alias_edge(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in sorted(self.graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.graph[dst][dst_nbr].get('weight', 1) / self.p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.graph[dst][dst_nbr].get('weight', 1))
            else:
                unnormalized_probs.append(self.graph[dst][dst_nbr].get('weight', 1) / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return normalized_probs

    def node2vec_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    alias_probs = self.get_alias_edge(prev, cur)
                    next_step = random.choices(cur_nbrs, weights=alias_probs, k=1)[0]
                    walk.append(next_step)
            else:
                break
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(node))
        return walks

    def train_model(self, walks):
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers, epochs=self.epochs)
        return model