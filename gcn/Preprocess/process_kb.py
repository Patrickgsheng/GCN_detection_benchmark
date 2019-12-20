from load_unprocessed_graph import *
from select_train_nodes import *
from combine_test_part import *

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('keep_train_tratio',0.1,'the training nodes keep percentage')
flags.DEFINE_string('destination_dir','/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/','The output destination dir')
flags.DEFINE_string('datasetname','kb_error_09_GCNinduce','induced graph name')
flags.DEFINE_float('edge_removal_ratio',0.1,'the edge removal ratio')

train_prefix = '/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/kb/kb_error_09'

def main():

    train_data = load_unprocessed_graph(train_prefix)
    print("Done loading training data..")
    graph_process=select_train_nodes(train_data)
    print("train node selection is finished")
    combine_test_part(graph_process,train_data)


if __name__ == '__main__':
    main()