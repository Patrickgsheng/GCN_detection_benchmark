import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import os
from matplotlib import pyplot as plt

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_cora_data(path,prefix, normalize=True):
    G_data = json.load(open(path + prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    # change graph adjacency matrix to sparse matrix format
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(G.adj))
    print("The number of edges")
    edge_num = G.number_of_edges()
    print(edge_num)
    print("The number of nodes")
    nodes_num = G.number_of_nodes()
    print(nodes_num)

    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(path+prefix + "-feats.npy"):
        feats = np.load(path+prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(path+prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    # just print the id_map keys range:
    # id_map_range = np.sort(id_map.keys())
    walks = []
    class_map = json.load(open(path + prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    # just print the class_map keys range:
    class_map_int_list = []
    for j in class_map.keys():
        class_map_int_list.append(int(j))
    class_map_range = np.sort(class_map_int_list)

    # generate y_train, y_val, y_test ndarray
    y_train = np.array([0, 0, 0, 0, 0, 0, 0])
    y_val = np.array([0, 0, 0, 0, 0, 0, 0])
    y_test = np.array([0, 0, 0, 0, 0, 0, 0])
    idx_train = range(140)
    idx_val = []
    idx_test = []
    for node in G.nodes():
        if node in idx_train:
            print("Train,currrent n is %d" % node)
            train_label = G.node[node]['labels']
            train_label = np.array(train_label)
            y_train = np.vstack((y_train, train_label))
            y_val = np.vstack((y_val, [0, 0, 0, 0, 0, 0, 0]))
            y_test = np.vstack((y_test, [0, 0, 0, 0, 0, 0, 0]))
        elif G.node[node]['test'] == False and G.node[node]['val'] == False:
            print("no label id,currrent n is %d" % node)
            y_train = np.vstack((y_train, [0, 0, 0, 0, 0, 0, 0]))
            y_val = np.vstack((y_val, [0, 0, 0, 0, 0, 0, 0]))
            y_test = np.vstack((y_test, [0, 0, 0, 0, 0, 0, 0]))
        elif G.node[node]['test'] == False and G.node[node]['val'] == True:
            print("Validation, current n is %d" % node)
            validation_label = G.node[node]['labels']
            validation_label = np.array(validation_label)
            y_val = np.vstack((y_val, validation_label))
            y_train = np.vstack((y_train, [0, 0, 0, 0, 0, 0, 0]))
            y_test = np.vstack((y_test, [0, 0, 0, 0, 0, 0, 0]))
            idx_val.append(node)
        elif G.node[node]['test'] == True and G.node[node]['val'] == False:
            print("Test, current n is %d" % node)
            test_label = G.node[node]['labels']
            test_label = np.array(test_label)
            y_test = np.vstack((y_test, test_label))
            y_train = np.vstack((y_train, [0, 0, 0, 0, 0, 0, 0]))
            y_val = np.vstack((y_val, [0, 0, 0, 0, 0, 0, 0]))
            idx_test.append(node)

    print("training label shape is")
    print(y_train.shape)
    y_train = np.delete(y_train, 0, axis=0)
    y_val = np.delete(y_val, 0, axis=0)
    y_test = np.delete(y_test, 0, axis=0)

    # generate train_mask, val_mask and test_mask
    train_mask = sample_mask(idx_train, len(G.node))
    val_mask = sample_mask(idx_val, len(G.node))
    test_mask = sample_mask(idx_test, len(G.node))

    # check how many train_mask is true:
    train_true_num = np.count_nonzero(train_mask)
    # Similarly for val_mask, test_mask
    val_true_num = np.count_nonzero(val_mask)
    test_true_num = np.count_nonzero(test_mask)


    node_degrees = list(G.degree().values())
    print("the maximum degree of the graph is %d" % max(node_degrees))

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    # add the train_removed Flag for each edge in G.edges
    # temp_useful_edges =0
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
            # temp_useful_edges+=1
    # print (G.node[edge[0]])
    # print ("The real edges that are taken account in is %d" %(temp_useful_edges))
    # 1432 useful edges marked with train_removed = False

    ''' Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean 
    and standard deviation are then stored to be used on later data using the transform method. 
    If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn 
    from other features correctly as expected.
    '''
    '''
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feats)
        feats = scaler.transform(feats)
    '''

    feats = sp.csr_matrix(feats)

    '''
    # visualize the graph
    options = {
        'arrows': True,
        'node_color': 'blue',
        'node_size': .05,
        'line_color': 'black',
        'linewidths': 1,
        'width': 0.1,
        'with_labels': False,
        'node_shape': '.',
        'node_list': range(G.number_of_nodes())
    }
    nx.draw_networkx(G, **options)
    #plt.savefig('/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/' + '/vis.png', dpi=1024)
    '''
    # print the diameter(maximum distance) of G
    '''
    k = nx.connected_component_subgraphs(G)
    diameter_list = []
    for i in k:
        print("Nodes in compoent.", i.nodes())
        diameter_list.append(nx.diameter(i))
    '''
    return adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask
