import tensorflow as tf
import numpy as np
import networkx as nx
from dumpJSON import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def select_train_nodes(origin_graph):

    G= origin_graph[0]
    feats = origin_graph[1]
    id_map = origin_graph[2]
    class_map = origin_graph[4]
    num_train_origin = len(origin_graph[5])
    train_idx = origin_graph[5]
    keep_train_num = int(num_train_origin*FLAGS.keep_train_tratio)

    temp_useful_edges = len(G.edges())

    print ("The real edges that are taken account in is %d" % (temp_useful_edges))

    #Generate a copy of original graph and only keep the training part of this copy graph
    G_temp = G.copy()
    for n in list(G_temp):
        if n not in train_idx:
            G_temp.remove_node(n)

    k = nx.connected_component_subgraphs(G_temp)
    dict = {}
    for i in k:
        print("Nodes in compoent.", i.nodes())
        if dict.get(nx.diameter(i)) == None:
            dict[nx.diameter(i)] = i.nodes()
        else:
            for node in i.nodes():
                dict[nx.diameter(i)].append(node)

    # sort the nx.diameter(i) and add the corresponding node set to the candidate_train_idx
    candidate_train_idx = []
    for i in sorted(dict, reverse=True):
        print((i, dict[i]))
        if (len(candidate_train_idx) <= keep_train_num):
            for j in dict[i]:
                candidate_train_idx.append(j)
            if len(candidate_train_idx) > keep_train_num:
                for j in range(len(dict[i])):
                    candidate_train_idx.pop(-1)
                remain_num = keep_train_num - len(candidate_train_idx)
                print("the remain_num is %d" % remain_num)
                last_idx_train = np.random.choice(dict[i], remain_num, replace=False)
                for j in last_idx_train:
                    candidate_train_idx.append(j)
    print("The selected training nodes length %d" % len(candidate_train_idx))
    assert (len(candidate_train_idx) == keep_train_num)

    keep_idx_train = candidate_train_idx

    keep_idx_train.sort()
    print("The number of kept training nodes is %d" % (len(keep_idx_train)))



    # Iterate through each node index in the training graph
    for i in list(G):
        if i in train_idx:
        # if the node is not in the "keep" list, remove it from the three objects
            if i not in keep_idx_train:
                G.node[i]['label']=None
                class_map[i]=None
        else:
            G.remove_node(i)
            id_map.pop(i)
            class_map.pop(i)
    print("The length of processed G is")
    print(len(G.node))
    print("The edge number is ")
    print(len(G.edges()))
    #only keep the training feats
    featsN = feats[train_idx,:]
    print("The feats shape is")
    print(featsN.shape)

    # save the training node id selection, from the original graph, what nodes are kept is clear
    #np.savetxt(FLAGS.destination_dir + '/' + FLAGS.datasetname + 'train_id_selection',
               #keep_idx_train, delimiter=",")

    print("trim process is finished!!!!")
    #dumpJSON(FLAGS.destination_dir, FLAGS.datasetname, G, id_map, class_map, featsN)

    return G
