from __future__ import division
from __future__ import print_function

import time
import tensorflow   as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from gcn.utils import *
from gcn.models import GCN, MLP
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from gcn.load_cora_data import *


#from sklearn.metrics import precision_recall_fscore_support as score


# Set random seed
#seed = 123
seed= 321
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 0, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
#flags.DEFINE_float('pollute_ratio',0.2,'the ratio of pollution')
#flags.DEFINE_float('attribute_pollution_ratio',0.04, 'the ratio of polluted attributes')

flags.DEFINE_integer('hidden0', 128, 'Number of units in hidden layer 0.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 0.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 0.')
#Need to be changed by dataset
#flags.DEFINE_integer('total_instance', 2708, 'Number of instances in this dataset')
flags.DEFINE_string('datasetname', 'cora_process', 'Dataset to be used (citeseer/cora).')
flags.DEFINE_string('datapath',"/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/cora/","Dataset path")

def calc_scores(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_accuracy_score=metrics.accuracy_score(y_true, y_pred,normalize=True)

    #print("accuracy is {:.5f}".format(balanced_accuracy_score))
    return precision[-1], recall[-1], fscore[-1], support[-1],balanced_accuracy_score




# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_bsbm_data(FLAGS.datapath,FLAGS.datasetname)
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_kb_data(FLAGS.datapath,FLAGS.datasetname)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_cora_data(FLAGS.datapath,FLAGS.datasetname)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    #calculate A^
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
'''
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    val_preds=[]
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.recall,model.precision, model.f1_score], feed_dict=feed_dict_val)
    val_prediction = tf.nn.softmax(outs[-1])
    val_prediction = sess.run(val_prediction)
    val_preds.append(val_prediction)
    val_labels = tf.boolean_mask(labels, mask)
    val_pred = tf.boolean_mask(val_preds[0], mask)
    val_labels = sess.run(val_labels)
    val_pred = sess.run(val_pred)
    scores = calc_scores(val_labels, val_pred)
    #print("precision is:")
    #print(scores[0])
    #print("recall is")
    #print(scores[1])
    #return outs_val[0],outs_val[1], scores[0],scores[1],scores[2], (time.time() - t_test)
    return outs_val[0],outs_val[1],scores[0],scores[1], scores[2], scores[3], (time.time() - t_test)
'''
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)



# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    #print(y_train.shape)
    #print(train_mask.shape)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})


    # Training step
    '''
    train_labels=y_train
    train_preds = []
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.precision, model.TP, model.FP,model.outputs], feed_dict=feed_dict)
    train_prediction = tf.nn.softmax(outs[-1])
    train_prediction = sess.run(train_prediction)
    train_preds.append(train_prediction)
    train_labels = tf.boolean_mask(train_labels, train_mask)
    train_pred = tf.boolean_mask(train_preds[0], train_mask)
    train_labels =sess.run(train_labels)
    train_pred = sess.run(train_pred)
    scores = calc_scores(train_labels, train_pred)
    #print("precision is:")
    #print(scores[0])
    '''
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)


    # Validation
    '''
    cost, acc, precision, recall, f1_score, val_support, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    '''
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)


    # Print results
    '''
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          #"train_acc=", "{:.5f}".format(outs[2]),
          #"train_precision=","{:.5f}".format(outs[3]),
          #"val_loss=", "{:.5f}".format(cost),
          #"val_acc=", "{:.5f}".format(acc),
          #"val_recall=", "{:.5f}".format(recall),
          #"val_precision=", "{:.5f}".format(precision),
          #"val_f1_score=", "{:.5f}".format(f1_score),
          "val_support=","{:.0f}".format(val_support),
          "time=", "{:.5f}".format(time.time() - t))

    test_cost, test_acc, test_precision, test_recall, test_f1_score, test_support, test_duration = evaluate(features, support, y_test,
                                                                                              test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "Test_accuracy=", "{:.5f}".format(test_acc),
          "Test_recall=", "{:.5f}".format(test_recall),
          "Test_precision=", "{:.5f}".format(test_precision),
          "Test_f1_score=", "{:.5f}".format(test_f1_score),
          #"Test_support=","{:.0f}".format(test_support),
          "time=", "{:.5f}".format(test_duration))
    '''
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))


    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
'''
test_cost, test_acc, test_precision, test_recall, test_f1_score,test_support, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc),
      "recall=", "{:.5f}".format(test_recall),
      "precision=", "{:.5f}".format(test_precision),
      "f1_score=","{:.5f}".format(test_f1_score),
      "time=", "{:.5f}".format(test_duration))
'''
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

#feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
#classification = sess.run(model.outputs, feed_dict= feed_dict_val)

#print (classification)

'''
####################################Some sample codes just for debugging####################
with tf.Session() as sess:
    labels_sam = np.array([[0, 1],
                           [0, 1],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0],
                           [1, 0]], dtype=np.float32)
    logits_sam = np.array([[0.02, 0.98],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.02, 0.98],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02],
                           [0.98, 0.02]], dtype=np.float32)
    y_true = np.argmax(labels_sam, axis=1)
    y_pred = np.argmax(logits_sam, axis=1)
    mask = [False,False,False,False,False,
            True, True, True, True, True, True, True,True, True,True]

    y_true_labels = tf.boolean_mask(y_true, mask)
    y_pred_labels = tf.boolean_mask(y_pred, mask)

    # precision, recall, fscore, support = metrics.precision_recall_fscore_support(labels_sam, logits_sam, average=None)
    y_true_labels=sess.run(y_true_labels)
    y_pred_labels= sess.run(y_pred_labels)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true_labels, y_pred_labels, average=None)
    balanced_accuracy_score = metrics.accuracy_score(y_true_labels, y_pred_labels, normalize=True)

    # print("accuracy is {:.5f}".format(balanced_accuracy_score))
    # fscore[-1], support[-1],balanced_accuracy_score

    print(precision[-1])
    print(recall[-1])
    print(fscore[-1])
    print(support[-1])
    print(balanced_accuracy_score)
'''


