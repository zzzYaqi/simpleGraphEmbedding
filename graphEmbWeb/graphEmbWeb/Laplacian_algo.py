from sklearn.manifold import SpectralEmbedding
import networkx as nx


class laplacian:
    def __init__(self, graph, n_components, random_state):
        self.graph = graph
        self.n_components = n_components
        self.random_state = random_state

    def train(self):
        matrix = nx.to_numpy_array(self.graph)
        embedding = SpectralEmbedding(n_components=self.n_components, affinity='precomputed',
                                      random_state=self.random_state)
        result_embeddings = embedding.fit_transform(matrix)
        return result_embeddings
