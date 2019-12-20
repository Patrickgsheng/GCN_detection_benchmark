import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from gcn.input_data import pollute_data
import json
import os
from networkx.readwrite import json_graph as jg


import sys
sys.path.insert(1, '/Users/april/Downloads/GCN_detection_benchmarkFinal/GCN_detection_benchmark/gcn/Preprocessing/')



def create_G_idM_classM(adjacency, features, testMask, valMask, labels):
    # 1. Create Graph
    print("Creating graph...")
    # Create graph from adjacency matrix
    G = nx.from_numpy_matrix(adjacency)
    num_nodes = G.number_of_nodes()

    # Change labels to int from numpy.int64
    labels = labels.tolist()
    for arr in labels:
        for integer in arr:
            integer = int(integer)

    # Iterate through each node, adding the features
    i = 0
    for n in list(G):
        G.node[i]['feature'] = list(map(float, list(features[i])))
        G.node[i]['test'] = bool(testMask[i])
        G.node[i]['val'] = bool(valMask[i])
        G.node[i]['labels'] = list(map(int, list(labels[i])))
        i += 1

    # 2. Create id-Map and class-Map
    print("Creating id-Map and class-Map...")
    # Initialize the dictionarys
    idM = {}
    classM = {}

    # Populate the dictionarys
    i = 0
    while i < num_nodes:
        idStr = str(i)
        idM[idStr] = i
        classM[idStr] = list(labels[i])
        i += 1

    return G, idM, classM





def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    #Use mask to translate a fully supervised setting to a semi-supervised setting
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    #combine all training and testing features as sparse matrix
    features = sp.vstack((allx, tx)).tolil()
    #change the testing features' order, the testing instances will follow training instances
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #change graph adjacency matrix to sparse matrix format
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #correspondingly adjust testing labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #attributes, labels = pollute_data_2(labels, features)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    #Just choose another 500 training instances as validation set
    idx_val = range(len(y), len(y)+500)

    '''
    idx_train = range(1208)
    idx_val = range(1208,  1208+ 500)


    attributes, labels = pollute_data(labels, features, idx_train, idx_val, idx_test)
    '''
    #testing the label rate of cora dataset
    if dataset_str == 'cora':
        num_train = len(y)
        total_num = len(ally)+len(ty)
        label_ratio_cora = num_train *1.0/total_num
        print(label_ratio_cora)

    if dataset_str == 'citeseer':
        num_train = len(y)
        total_num = len(ally) + len(ty)
        label_ratio_citeseer = num_train * 1.0 / total_num
        print(label_ratio_citeseer)


    #vector of size 2708, idx_train as true
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    #only assign label value when the train_mask as true
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    #testing instance starts from 1708
    y_test[test_mask, :] = labels[test_mask, :]


    #Translate adj to numpy arrays
    adj_np = adj.toarray()

    #translate features to numpy arrays
    features_np = features.toarray()

    #generate the graph and id_map, class_map
    G, IDMap, classMap =create_G_idM_classM(adj_np, features_np, test_mask, val_mask, labels)


    #at this stage, for all validation nodes, test nodes we have their labels but use mask tp make them
    #all [0 0 0 0 0 0 0]

    num_edges =len(G.edges())
    print(num_edges)
    print(G.number_of_edges())

    #Dump everything into .json files and one .npy
    if dataset_str == 'cora':
        graphFile_prefix = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/cora'
        dataset_name = 'cora_process'
        dumpJSON(graphFile_prefix, dataset_name, G, IDMap, classMap, features_np)

    if dataset_str == 'citeseer':
        graphFile_prefix = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/citeseer'
        dataset_name = 'citeseer_process'
        dumpJSON(graphFile_prefix, dataset_name, G, IDMap, classMap, features_np)


    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def load_bsbm_data(path,prefix, normalize=True):
    G_data = json.load(open(path+prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    # change graph adjacency matrix to sparse matrix format
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(G.adj))
    print("The number of edges")
    edge_num = G.number_of_edges()
    print(edge_num)
    print("The number of nodes")
    nodes_num = G.number_of_nodes()
    print(nodes_num)
    # print G.nodes()[0]
    # check G.nodes()[0] is an integer or not
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
    class_map = json.load(open(path+prefix + "-class_map.json"))
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

    #generate y_train, y_val, y_test ndarray
    y_train =np.array([0,0])
    y_val = np.array([0,0])
    y_test =np.array([0,0])
    idx_train =[]
    idx_val=[]
    idx_test=[]
    for node in G.nodes():
        if G.node[node]['test'] == False and G.node[node]['val']==False:
                print("Train,currrent n is %d" % node)
                train_label = G.node[node]['label']
                train_label = np.array(train_label)
                y_train = np.vstack((y_train, train_label))
                y_val = np.vstack((y_val,[0,0]))
                y_test = np.vstack((y_test,[0,0]))
                idx_train.append(node)
        elif G.node[node]['test'] == False and G.node[node]['val']==True:
                print("Validation, current n is %d" %node)
                validation_label = G.node[node]['label']
                validation_label = np.array(validation_label)
                y_val = np.vstack((y_val,validation_label))
                y_train = np.vstack((y_train, [0, 0]))
                y_test = np.vstack((y_test,[0,0]))
                idx_val.append(node)
        elif G.node[node]['test'] == True and G.node[node]['val']==False:
                print("Test, current n is %d" %node)
                test_label = G.node[node]['label']
                test_label = np.array(test_label)
                y_test = np.vstack((y_test,test_label))
                y_train = np.vstack((y_train, [0, 0]))
                y_val = np.vstack((y_val, [0, 0]))
                idx_test.append(node)

    print("training label shape is")
    #print(y_train.shape)
    y_train = np.delete(y_train,0,axis=0)
    y_val = np.delete(y_val,0,axis=0)
    y_test = np.delete(y_test,0,axis=0)
    print(y_train.shape)

    #generate train_mask, val_mask and test_mask
    train_mask = sample_mask(idx_train, len(G.node))
    val_mask = sample_mask(idx_val, len(G.node))
    test_mask = sample_mask(idx_test, len(G.node))

    #check how many train_mask is true:
    train_true_num = np.count_nonzero(train_mask)
    #Similarly for val_mask, test_mask
    val_true_num = np.count_nonzero(val_mask)
    test_true_num = np.count_nonzero(test_mask)

    # print the anormaly ground truth number
    anormaly_count_gt = 0
    anormaly_count_vl = 0
    anormaly_count_tn = 0
    for node in G.nodes():
        if G.node[node]['test'] == True:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_gt += 1
        if G.node[node]['val'] == True:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_vl += 1
        if G.node[node]['val'] != True and G.node[node]['test'] != True:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_tn += 1
    print("anormaly in test data is %d" % (anormaly_count_gt))
    print("anormaly in validation data is %d" % (anormaly_count_vl))
    print("anormaly in training data is %d" % (anormaly_count_tn))

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
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)


    feats = sp.csr_matrix(feats)
    return adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask


def dumpJSON(destDirect, datasetName, graph, idMap, classMap, features):
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)
    # print(graph.number_of_edges())
    # Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'
    json_ID_name = destDirect + '/' + datasetName + '-id_map.json'
    json_C_name = destDirect + '/' + datasetName + '-class_map.json'
    npy_F_name = destDirect + '/' + datasetName + '-feats'

    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)

    # Dump idMap into json file
    with open(json_ID_name, 'w') as outputFile:
        json.dump(idMap, outputFile)

    # Dump classMap into json file
    with open(json_C_name, 'w') as outputFile:
        json.dump(classMap, outputFile)

    # Save features as .npy file
    print("Saving features as numpy file...")
    np.save(npy_F_name, features)

    print("all part finished")

