import matplotlib.pyplot as plt
import networkx as nx
from graphEmbWeb.graphEmbWeb.gameGraph import gameGraph


graph = gameGraph()
G = graph.get_graph()

# 计算节点的布局
pos = nx.spring_layout(G)

# 绘制图的节点和边
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15)

# 显示图形
plt.show()

plt.show()

