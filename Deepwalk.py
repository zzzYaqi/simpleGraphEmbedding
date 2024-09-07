from gensim.models import Word2Vec
import random
from graphEmbWeb.graphEmbWeb.gameGraph import gameGraph
from graphEmbWeb.graphEmbWeb.userGraph import userGraph
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# 1.返回一个walk结果的list

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, dimensions, window_size, epochs):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = 2
        self.epochs = epochs

    def get_result(self):
        # 生成随机游走并训练模型
        walks = deepwalk.generate_walks()
        model = deepwalk.train_model(walks)
        embeddings = model.wv

        # 提取嵌入向量
        embedding_vectors = [embeddings[str(node)] for node in graph.nodes()]
        # 将 embedding_vectors 转换为 numpy 数组
        embedding_vectors_np = np.array(embedding_vectors)
        return embedding_vectors_np

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


# 示例：使用 DeepWalk 算法

# 创建一个图
G = userGraph()
graph = G.get_graph()

# 初始化 DeepWalk
deepwalk = DeepWalk(graph=graph, walk_length=10, num_walks=5, dimensions=48, window_size=5, epochs=20)

# 生成随机游走并训练模型
walks = deepwalk.generate_walks()
model = deepwalk.train_model(walks)
embeddings = model.wv

# 提取嵌入向量
embedding_vectors = [embeddings[str(node)] for node in graph.nodes()]
# 将 embedding_vectors 转换为 numpy 数组
embedding_vectors_np = np.array(embedding_vectors)


# 使用t-SNE进行降维
# tsne = TSNE(n_components=2, random_state=7)
#embeddings_2d = tsne.fit_transform(embedding_vectors_np)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embedding_vectors_np)
nodes_data = [
    {'id': node_id, 'x': coords[0], 'y': coords[1]}
    for node_id, coords in zip(range(0, len(graph.nodes)), embeddings_2d)
]
# 绘制结果
plt.figure(figsize=(10, 8))
for i, node in enumerate(graph.nodes()):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.annotate(node, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()
plt.show()
