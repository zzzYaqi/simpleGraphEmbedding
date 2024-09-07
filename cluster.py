import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For visualization
from graphEmbWeb.graphEmbWeb.Node2vec_algo import node2Vec
from graphEmbWeb.graphEmbWeb.Laplacian_algo import laplacian
from graphEmbWeb.graphEmbWeb.gameGraph import gameGraph
from graphEmbWeb.graphEmbWeb.userGraph import userGraph
from graphEmbWeb.graphEmbWeb.Deepwalk_algo import deepWalk
from sklearn.decomposition import PCA


def dw_embedding(graph, walk_length_1, num_walks_1, dimensions_1, window_size_1, epoches_1):
    dw = deepWalk(graph=graph, walk_length=walk_length_1, num_walks=num_walks_1, dimensions=dimensions_1,
                  window_size=window_size_1, epochs=epoches_1)

    walks = dw.generate_walks()
    deepwalks_dict = {'walks': walks}
    model = dw.train_model(walks)
    embeddings = model.wv
    # 提取嵌入向量
    embedding_vectors = [embeddings[str(node)] for node in graph.nodes()]
    embedding_vectors_np = np.array(embedding_vectors)
    return deepwalks_dict, embedding_vectors_np


def n2v_embedding(graph, walk_length_2, num_walks_2, dimensions_2, window_size_2, epoches_2, p_2, q_2):
    n2v = node2Vec(graph=graph, walk_length=walk_length_2, num_walks=num_walks_2, dimensions=dimensions_2,
                   window_size=window_size_2, epochs=epoches_2, p=p_2, q=q_2)
    walks = n2v.generate_walks()
    node2vec_dict = {'walks': walks}
    model = n2v.train_model(walks)
    embeddings = model.wv
    embedding_vectors = [embeddings[str(node)] for node in graph.nodes()]
    embedding_vectors_np = np.array(embedding_vectors)
    return node2vec_dict, embedding_vectors_np


def lap_embedding(graph, random_state, n_components, affinity):
    lap = laplacian(graph=graph, n_components=n_components, random_state=random_state, affinity=affinity)
    embedding_vectors_np = np.array(lap.train())
    return embedding_vectors_np


# values for deepwalk
walk_length_1 = 5
num_walks_1 = 30
dimensions_1 = 128
window_size_1 = 15
epoches_1 = 10

# values for node2vec
walk_length_2 = 15
num_walks_2 = 41
dimensions_2 = 32
window_size_2 = 15
epoches_2 = 50
p_2 = 0.25
q_2 = 0.5

# values for laplacian
n_components = 3
affinity = 'nearest_neighbors'
random_state = None

# Creating a random graph
G = gameGraph()
graph = G.get_graph()

_, node_embeddings_dw = dw_embedding(graph, walk_length_1, num_walks_1, dimensions_1, window_size_1, epoches_1)
_, embeddings_n2v = n2v_embedding(graph, walk_length_2, num_walks_2, dimensions_2,
                                      window_size_2, epoches_2, p_2, q_2)
embeddings_sp = lap_embedding(graph, random_state, n_components, affinity)

# Number of clusters
k = 6

# Running K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(embeddings_n2v)

# Getting the cluster labels assigned to each node
cluster_labels = kmeans.labels_

# Reduce dimensions to 2D for visualization, using t-SNE
tsne = PCA(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings_n2v)

# Plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Clusters of Graph Nodes by K-Means')
plt.show()

# Example: Print nodes in each cluster
for i in range(k):
    print(f"Cluster {i+1}: {np.where(cluster_labels == i)[0]}")

