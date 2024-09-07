import networkx as nx
from django.urls import reverse
from urllib.parse import urlencode

from .Node2vec_algo import node2Vec
from .Laplacian_algo import laplacian
from .gameGraph import gameGraph
from .userGraph import userGraph
from .Deepwalk_algo import deepWalk
from .User import user
from django.shortcuts import render, redirect

import json
from sklearn.decomposition import PCA
import numpy as np


def app_user(request):
    User = user()
    uid, ulike = User.return_user()
    # graph for page2
    u_graph = userGraph()
    u_data = u_graph.get_graph()
    user_graph_json = u_graph.return_data()
    user_node_group = u_graph.get_node_group()
    user_node_name = u_graph.get_node_name()
    # values for deepwalk
    walk_length_1 = 10
    num_walks_1 = 5
    dimensions_1 = 24
    window_size_1 = 5
    epoches_1 = 20

    # values for node2vec
    walk_length_2 = 10
    num_walks_2 = 5
    dimensions_2 = 24
    window_size_2 = 5
    epoches_2 = 20
    p_2 = 1
    q_2 = 1

    random_state = 42

    if request.method == 'POST':
        # 从 POST 请求中更新参数
        form_id = request.POST.get('form_id')
        params = {}
        if form_id == 'form_1_1':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'walk_length_1': request.POST.get('walkLength', 10),
                'numWalks_1': request.POST.get('numWalks', 5),
                'dimensions_1': request.POST.get('dimensions', 24),
                'windowSize_1': request.POST.get('windowSize', 5),
                'epoches_1': request.POST.get('epoches', 20),
            }

        elif form_id == 'form_1_2':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'walk_length_2': request.POST.get('walkLength', 10),
                'numWalks_2': request.POST.get('numWalks', 5),
                'dimensions_2': request.POST.get('dimensions', 24),
                'windowSize_2': request.POST.get('windowSize', 5),
                'epoches_2': request.POST.get('epoches', 20),
                'p_2': request.POST.get('p', 1),
                'q_2': request.POST.get('q', 1),
            }
        elif form_id == 'form_1_3':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'random_state': request.POST.get('random_state', 42),
            }

        return redirect(reverse('app_user') + '?' + urlencode(params))

    formId = str(request.GET.get('form', 'form'))
    if formId == 'form_1_1':
        # 从 GET 请求或使用默认值获取参数
        walk_length_1 = int(request.GET.get('walk_length_1', 10))
        num_walks_1 = int(request.GET.get('numWalks_1', 5))
        dimensions_1 = int(request.GET.get('dimensions_1', 24))
        window_size_1 = int(request.GET.get('windowSize_1', 5))
        epoches_1 = int(request.GET.get('epoches_1', 20))
    elif formId == 'form_1_2':
        walk_length_2 = int(request.GET.get('walk_length_2', 10))
        num_walks_2 = int(request.GET.get('numWalks_2', 5))
        dimensions_2 = int(request.GET.get('dimensions_2', 24))
        window_size_2 = int(request.GET.get('windowSize_2', 5))
        epoches_2 = int(request.GET.get('epoches_2', 20))
        p_2 = float(request.GET.get('p_2', 1))
        q_2 = float(request.GET.get('q_2', 1))
    elif formId == 'form_1_3':
        random_state = int(request.GET.get('random_state', 42))

    deepwalks_dict, embedding_vectors_dw = dw_embedding(u_data, walk_length_1, num_walks_1, dimensions_1, window_size_1,
                                                        epoches_1)
    node2vec_dict, embedding_vectors_n2v = n2v_embedding(u_data, walk_length_2, num_walks_2, dimensions_2,
                                                         window_size_2, epoches_2, p_2, q_2)
    embedding_vectors_lap = lap_embedding(u_data, random_state)
    pca_1 = PCA(n_components=2)
    embeddings_2d_dw = pca_1.fit_transform(embedding_vectors_dw)
    embedding_data_dw = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(u_data.nodes)), user_node_group, user_node_name, embeddings_2d_dw)
    ]

    pca_2 = PCA(n_components=2)
    embeddings_2d_n2v = pca_2.fit_transform(embedding_vectors_n2v)
    embedding_data_n2v = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(u_data.nodes)), user_node_group, user_node_name, embeddings_2d_n2v)
    ]

    embedding_data_lap = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(u_data.nodes)), user_node_group, user_node_name, embedding_vectors_lap)
    ]

    context = {
        'uid': uid,
        'ulike': ulike,
        'node_name': user_node_name,
        'graph_json': user_graph_json,
        'deepwalks_dict': json.dumps(deepwalks_dict),
        'node2vev_dict': json.dumps(node2vec_dict),
        'walk_length_1': walk_length_1,
        'num_walks_1': num_walks_1,
        'dimensions_1': dimensions_1,
        'window_size_1': window_size_1,
        'epoches_1': epoches_1,
        'walk_length_2': walk_length_2,
        'num_walks_2': num_walks_2,
        'dimensions_2': dimensions_2,
        'window_size_2': window_size_2,
        'epoches_2': epoches_2,
        'p_2': p_2,
        'q_2': q_2,
        'random_state':random_state,
        'embedding_data_dw': embedding_data_dw,
        'embedding_data_n2v': embedding_data_n2v,
        'embedding_data_lap':embedding_data_lap
    }

    return render(request, 'app_user.html', context)


def app_overview(request):
    User = user()
    uid, ulike = User.return_user()
    g_graph = gameGraph()
    game_graph_json = g_graph.return_data()
    game_node_name = g_graph.get_node_name()
    u_graph = userGraph()
    user_graph_json = u_graph.return_data()
    context = {
        'uid': uid,
        'ulike': ulike,
        'node_name': game_node_name,
        'game_graph_json': game_graph_json,
        'user_graph_json': user_graph_json
    }
    return render(request, 'app_overview.html', context)


def eva_overview(request):
    karate = nx.karate_club_graph()
    u_graph = nx.watts_strogatz_graph(n=50, k=6, p=0.04)
    game_graph_json = nx.json_graph.node_link_data(u_graph)
    user_graph_json = nx.json_graph.node_link_data(karate)
    context = {
        'game_graph_json': game_graph_json,
        'user_graph_json': user_graph_json
    }
    return render(request, 'eva_overview.html',context)


def app_game(request):
    User = user()
    uid, ulike = User.return_user()

    # graph for page2
    g_graph = gameGraph()
    g_data = g_graph.get_graph()
    game_graph_json = g_graph.return_data()
    game_node_group = g_graph.get_node_group()
    game_node_name = g_graph.get_node_name()

    # values for deepwalk
    walk_length_1 = 10
    num_walks_1 = 5
    dimensions_1 = 24
    window_size_1 = 5
    epoches_1 = 20

    # values for node2vec
    walk_length_2 = 10
    num_walks_2 = 5
    dimensions_2 = 24
    window_size_2 = 5
    epoches_2 = 20
    p_2 = 1
    q_2 = 1

    # values for laplacian
    random_state = 42

    if request.method == 'POST':
        # 从 POST 请求中更新参数
        form_id = request.POST.get('form_id')
        params = {}
        if form_id == 'form_1_1':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'walk_length_1': request.POST.get('walkLength', 10),
                'numWalks_1': request.POST.get('numWalks', 5),
                'dimensions_1': request.POST.get('dimensions', 24),
                'windowSize_1': request.POST.get('windowSize', 5),
                'epoches_1': request.POST.get('epoches', 20),
            }

        elif form_id == 'form_1_2':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'walk_length_2': request.POST.get('walkLength', 10),
                'numWalks_2': request.POST.get('numWalks', 5),
                'dimensions_2': request.POST.get('dimensions', 24),
                'windowSize_2': request.POST.get('windowSize', 5),
                'epoches_2': request.POST.get('epoches', 20),
                'p_2': request.POST.get('p', 1),
                'q_2': request.POST.get('q', 1),
            }
        elif form_id == 'form_1_3':
            params = {
                'form': request.POST.get('form_id', 'form'),
                'random_state': request.POST.get('random_state', 42),
            }
        return redirect(reverse('app_game') + '?' + urlencode(params))

    formId = str(request.GET.get('form', 'form'))
    if formId == 'form_1_1':
        # 从 GET 请求或使用默认值获取参数
        walk_length_1 = int(request.GET.get('walk_length_1', 10))
        num_walks_1 = int(request.GET.get('numWalks_1', 5))
        dimensions_1 = int(request.GET.get('dimensions_1', 24))
        window_size_1 = int(request.GET.get('windowSize_1', 5))
        epoches_1 = int(request.GET.get('epoches_1', 20))

    elif formId == 'form_1_2':
        walk_length_2 = int(request.GET.get('walk_length_2', 10))
        num_walks_2 = int(request.GET.get('numWalks_2', 5))
        dimensions_2 = int(request.GET.get('dimensions_2', 24))
        window_size_2 = int(request.GET.get('windowSize_2', 5))
        epoches_2 = int(request.GET.get('epoches_2', 20))
        p_2 = float(request.GET.get('p_2', 1))
        q_2 = float(request.GET.get('q_2', 1))

    elif formId == 'form_1_3':
        random_state = int(request.GET.get('random_state', 42))

    deepwalks_dict, embedding_vectors_dw = dw_embedding(g_data, walk_length_1, num_walks_1, dimensions_1, window_size_1,
                                                        epoches_1)
    node2vec_dict, embedding_vectors_n2v = n2v_embedding(g_data, walk_length_2, num_walks_2, dimensions_2,
                                                         window_size_2, epoches_2, p_2, q_2)
    embedding_vectors_lap = lap_embedding(g_data,random_state)

    pca_1 = PCA(n_components=2)
    embeddings_2d_dw = pca_1.fit_transform(embedding_vectors_dw)
    embedding_data_dw = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(g_data.nodes)), game_node_group, game_node_name, embeddings_2d_dw)
    ]

    pca_2 = PCA(n_components=2)
    embeddings_2d_n2v = pca_2.fit_transform(embedding_vectors_n2v)
    embedding_data_n2v = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(g_data.nodes)), game_node_group, game_node_name, embeddings_2d_n2v)
    ]

    embedding_data_lap = [
        {'id': node_id, 'group': group, 'name': name, 'coord': [coords[0], coords[1]]}
        for node_id, group, name, coords in
        zip(range(0, len(g_data.nodes)), game_node_group, game_node_name, embedding_vectors_lap)
    ]
    context = {
        'uid': uid,
        'ulike': ulike,
        'node_name': game_node_name,
        'graph_json': game_graph_json,
        'deepwalks_dict': json.dumps(deepwalks_dict),
        'node2vev_dict': json.dumps(node2vec_dict),
        'walk_length_1': walk_length_1,
        'num_walks_1': num_walks_1,
        'dimensions_1': dimensions_1,
        'window_size_1': window_size_1,
        'epoches_1': epoches_1,
        'walk_length_2': walk_length_2,
        'num_walks_2': num_walks_2,
        'dimensions_2': dimensions_2,
        'window_size_2': window_size_2,
        'epoches_2': epoches_2,
        'p_2': p_2,
        'q_2': q_2,
        'random_state':random_state,
        'embedding_data_dw': embedding_data_dw,
        'embedding_data_n2v': embedding_data_n2v,
        'embedding_data_lap':embedding_data_lap
    }

    return render(request, 'app_game.html', context)


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


def lap_embedding(graph, random_state):
    lap = laplacian(graph=graph, n_components=2, random_state=random_state)
    embedding_vectors_np = np.array(lap.train())
    return embedding_vectors_np

