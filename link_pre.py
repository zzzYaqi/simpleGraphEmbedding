import networkx as nx
import numpy as np
from networkx.drawing.tests.test_pylab import plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_curve
import pandas as pd
from graphEmbWeb.graphEmbWeb.Node2vec_algo import node2Vec
from graphEmbWeb.graphEmbWeb.Laplacian_algo import laplacian
from graphEmbWeb.graphEmbWeb.gameGraph import gameGraph
from graphEmbWeb.graphEmbWeb.userGraph import userGraph
from graphEmbWeb.graphEmbWeb.Deepwalk_algo import deepWalk
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score


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


def get_embeddings_by_method(graph, method_id):
    if method_id == 'DeepWalk':
        _, embeddings = dw_embedding(graph, walk_length_1, num_walks_1, dimensions_1, window_size_1, epoches_1)

    elif method_id == 'Node2Vec':
        _, embeddings = n2v_embedding(graph, walk_length_2, num_walks_2, dimensions_2,
                                      window_size_2, epoches_2, p_2, q_2)
    else:
        embeddings = lap_embedding(graph, random_state, n_components, affinity)

    return embeddings

# Function to calculate cosine similarity between two nodes
def calculate_similarity(node1, node2, embeddings):
    return cosine_similarity([embeddings[node1]], [embeddings[node2]])[0][0]

import random

G = userGraph()
graph = G.get_graph()

potential_links = list(nx.non_edges(graph))

# Generate random embeddings for demonstration; replace this with your actual method
_, node_embeddings_dw = dw_embedding(graph, walk_length_1, num_walks_1, dimensions_1, window_size_1, epoches_1)
_, embeddings_n2v = n2v_embedding(graph, walk_length_2, num_walks_2, dimensions_2,
                                      window_size_2, epoches_2, p_2, q_2)
embeddings_sp = lap_embedding(graph, random_state, n_components, affinity)


# Generate predictions (similarity scores) for each potential link
predictions_dw = [calculate_similarity(u, v, node_embeddings_dw) for u, v in potential_links]
predictions_n2v = [calculate_similarity(u, v, embeddings_n2v) for u, v in potential_links]
predictions_sp = [calculate_similarity(u, v, embeddings_sp) for u, v in potential_links]

true_labels = [1 if random.random() < 0.1 else 0 for _ in potential_links]

# Threshold predictions to get binary labels for precision calculation
predicted_labels_dw = [1 if score > 0.5 else 0 for score in predictions_dw]
predicted_labels_n2v = [1 if score > 0.5 else 0 for score in predictions_n2v]
predicted_labels_sp = [1 if score > 0.5 else 0 for score in predictions_sp]

# Calculate ROC AUC Score

roc_auc_dw = roc_auc_score(true_labels, predicted_labels_dw)
roc_auc_n2v = roc_auc_score(true_labels, predicted_labels_n2v)
roc_auc_sp = roc_auc_score(true_labels, predicted_labels_sp)

# Calculate Precision Score
precision_sp = precision_score(true_labels, predicted_labels_sp)
precision_dw = precision_score(true_labels, predicted_labels_dw)
precision_n2v = precision_score(true_labels, predicted_labels_n2v)


results = []
results.append({'Algorithm': f'Algorithm Spectral', 'Mean AUC': roc_auc_sp, 'Mean Precision': precision_sp})
results.append({'Algorithm': f'Algorithm DeepWalk', 'Mean AUC': roc_auc_dw, 'Mean Precision': precision_dw})
results.append({'Algorithm': f'Algorithm Node2Vec', 'Mean AUC': roc_auc_n2v, 'Mean Precision': precision_n2v})
results_df = pd.DataFrame(results)
print(results_df)

fpr, tpr, thresholds = roc_curve(true_labels, precision_sp)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#
# # Evaluate all algorithms
# results = []
# for method_id in ['Spectral', 'DeepWalk', 'Node2Vec']:
#     mean_auc, mean_precision = perform_cross_validation(graph, potential_links, method_id)
#     results.append({'Algorithm': f'Algorithm {method_id}', 'Mean AUC': mean_auc, 'Mean Precision': mean_precision})
#
# results_df = pd.DataFrame(results)
# print(results_df)

#
# _,embeddings_deepwalk = dw_embedding(graph, walk_length_1, num_walks_1, dimensions_1, window_size_1,
#                                                         epoches_1)
# _,embeddings_node2vec = n2v_embedding(graph, walk_length_2, num_walks_2, dimensions_2,
#                                                          window_size_2, epoches_2, p_2, q_2)
# embeddings_spectral = lap_embedding(graph,random_state)
#
# adj_matrix = nx.adjacency_matrix(graph)
# actual_links = adj_matrix.toarray()
# # 真实的链接情况，1表示节点对连接，0表示不连接（这也是模拟数据）
#
#
# # 计算相似度矩阵
# similarity_deepwalk = cosine_similarity(embeddings_deepwalk)
# similarity_node2vec = cosine_similarity(embeddings_node2vec)
# similarity_spectral = cosine_similarity(embeddings_spectral)
#
# # 设置相似度阈值
# threshold = 0.5
#
# # 预测链接存在
# predicted_links_deepwalk = (similarity_deepwalk > threshold).astype(int)
# predicted_links_node2vec = (similarity_node2vec > threshold).astype(int)
# predicted_links_spectral = (similarity_spectral > threshold).astype(int)
#
# # 计算准确率
# accuracy_deepwalk = accuracy_score(actual_links.ravel(), predicted_links_deepwalk.ravel())
# accuracy_node2vec = accuracy_score(actual_links.ravel(), predicted_links_node2vec.ravel())
# accuracy_spectral = accuracy_score(actual_links.ravel(), predicted_links_spectral.ravel())
#
# # 创建结果表格
# results = pd.DataFrame({
#     'Model': ['DeepWalk', 'Node2Vec', 'Spectral'],
#     'Accuracy': [accuracy_deepwalk, accuracy_node2vec, accuracy_spectral]
# })
#
# # 打印表格
# print(results)


# def evaluate_model(embedding):
#     threshold = 0.5
#     similarity_node2vec = cosine_similarity(embedding)
#     predicted_links_node2vec = (similarity_node2vec > threshold).astype(int)
#     return predicted_links_node2vec
#
# G = userGraph()
# graph = G.get_graph()
#
# adj_matrix = nx.adjacency_matrix(graph)
# actual_links = adj_matrix.toarray()
#
# import random
#
# num_iterations = 100
# best_accuracy = 0
# best_params = {}
#
# n_components_options = [2, 3, 5, 10]
# affinity_options = ['nearest_neighbors', 'rbf']
# random_state_options = [42, None]
#
# for _ in range(num_iterations):
#
#     n_components = random.choice(n_components_options)
#     affinity = random.choice(affinity_options)
#     random_state = random.choice(random_state_options)
#     embeddings_spectral = lap_embedding(graph, random_state,n_components, affinity)
#
#     predicted_links_node2vec = evaluate_model(embeddings_spectral)
#     accuracy = accuracy_score(actual_links.ravel(), predicted_links_node2vec.ravel())
#     params = {
#         'nc':n_components,
#         'aff':affinity,
#         'rs':random_state
#     }
#     print(params)
#     print(accuracy)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_params = params
#
# print('Best Accuracy:', best_accuracy)
# print('Best Parameters:', best_params)
