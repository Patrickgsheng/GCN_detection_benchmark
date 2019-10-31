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


#from sklearn.metrics import precision_recall_fscore_support as score


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 150, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('pollute_ratio',0.2,'the ratio of pollution')
flags.DEFINE_float('attribute_pollution_ratio',0.04, 'the ratio of polluted attributes')
flags.DEFINE_integer('hidden0', 32, 'Number of units in hidden layer 0.')
#flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 0.')
#flags.DEFINE_integer('hidden3', 8, 'Number of units in hidden layer 0.')
#Need to be changed by dataset
flags.DEFINE_integer('total_instance', 2708, 'Number of instances in this dataset')
flags.DEFINE_string('datasetname', 'bsbm_100_error03', 'Dataset to be used (citeseer/cora).')
flags.DEFINE_string('datapath',"/Users/april/Downloads/GraphSAGE_Benchmark-master/processed/BSBM/","Dataset path")

def calc_scores(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_accuracy_score=metrics.accuracy_score(y_true, y_pred,normalize=True)

    #print("accuracy is {:.5f}".format(balanced_accuracy_score))
    return precision[-1], recall[-1], fscore[-1], support[-1],balanced_accuracy_score




# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_bsbm_data(FLAGS.datapath,FLAGS.datasetname)

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
    return outs_val[0],outs_val[1], scores[0],scores[1],scores[2], (time.time() - t_test)


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

    # Validation
    cost, acc, precision, recall, f1_score, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]),
          "train_precision=","{:.5f}".format(outs[3]),
          "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc),
          "val_recall=", "{:.5f}".format(recall),
          "val_precision=", "{:.5f}".format(precision),
          "val_f1_score=", "{:.5f}".format(f1_score),
          "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_precision, test_recall, test_f1_score,test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc),
      "recall=", "{:.5f}".format(test_recall),
      "precision=", "{:.5f}".format(test_precision),
      "f1_score=","{:.5f}".format(test_f1_score),
      "time=", "{:.5f}".format(test_duration))

#feed_dict_val = construct_feed_dict(features, support, y_test, test_mask, placeholders)
#classification = sess.run(model.outputs, feed_dict= feed_dict_val)

#print (classification)

'''
####################################Some sample codes just for debugging####################
labels_sam = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
logits_sam = np.array([[1, 2, 7],
                   [2, 1, 3],
                   [6, 1, 3],
                   [8, 2, 0],
                   [3, 6, 1]], dtype=np.float32)

mask =np.array([False,True,False,True,True])
num_classes = labels_sam.shape[1]
y_pred = tf.argmax(logits_sam,1)
y_true = tf.argmax(labels_sam,1)
correct_prediction = tf.equal(y_pred,y_true)
accuracy_all =tf.cast(correct_prediction,tf.float32)
mask = tf.cast(mask, dtype=tf.float32)
mask /= tf.reduce_mean(mask)
accuracy_all *=mask
accuracy_mean = tf.reduce_mean(accuracy_all)
sess.run(accuracy_mean )
sess.run(accuracy_all)


labels_sam = np.array([[0, 1],
                   [0, 1],
                   [1, 0],
                   [1, 0],
                   [1, 0]], dtype=np.float32)
logits_sam = np.array([[1, 2],
                   [2, 1],
                   [6, 1],
                   [8, 2],
                   [6, 3]], dtype=np.float32)


mask =np.array([False,True,True,True,True])
y_true = tf.argmax(labels_sam,1)
y_pred = tf.argmax(logits_sam,1)
y_true = tf.boolean_mask(y_true, mask)
y_pred = tf.boolean_mask(y_pred, mask)
with tf.Session() as sess: print(y_true.eval())
with tf.Session() as sess: print(y_pred.eval())

TP = tf.count_nonzero((y_pred-1)*(y_true-1))
TN = tf.count_nonzero(y_pred * y_true)
FP = tf.count_nonzero((y_pred-1)*y_true)
FN = tf.count_nonzero(y_pred *(y_true-1))

############## The TP means the dirty instance identification ###############
TP = tf.count_nonzero(y_pred * y_true)
TN = tf.count_nonzero((y_pred-1)*(y_true-1))
FN = tf.count_nonzero(y_pred *(y_true-1))
FP = tf.count_nonzero((y_pred-1)*y_true)
#####################################################
with tf.Session() as sess: print(TP)
with tf.Session() as sess: print(TN)
sess.run(FP)
sess.run(FN)


precision = tf.divide(TP, TP+FP)
sess.run(precision)
sess.run(y_true)
sess.run(y_pred)


y_pred = tf.nn.softmax(logits_sam)
sess.run(y_pred)
y_true = np.argmax(labels_sam,1)
y_pred = np.argmax(logits_sam,1)
precision=tf.metrics.precision(y_true,y_pred,mask)
pred_p = (y_pred > 0).sum()

precision = precision_score(labels_sam, y_pred, average='macro')
sess.run(precision[0])



precision, update_op = tf.metrics.precision_at_k(y_true, logits_sam,1,class_id=2)
sess.run(tf.local_variables_initializer())

precision = sess.run(precision)
print(precision)



loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_sam, labels=labels_sam)
sess.run(loss)
mask =np.array([False,True,True,True,True])
mask = tf.cast(mask, dtype=tf.float32)
mask /= tf.reduce_mean(mask)
sess.run(mask)
loss *= mask
a= tf.shape(mask)
# your class weights
mask = mask.reshape(mask.size, 1)
tf.reshape(mask, [labels_sam.shape[0],1])

mask_1 =tf.broadcast_to(mask,[labels_sam.shape[1],5])
mask_1 =tf.transpose(mask_1)
sess.run(mask_1)
labels_sum = tf.reduce_sum(labels_sam*mask_1,0)
sess.run(labels_sum)
x_max = tf.reduce_max(labels_sum)
sess.run(x_max)
class_weights = x_max/labels_sum
sess.run(class_weights)
# deduce weights for batch samples based on their true label
weights_test = tf.reduce_sum(class_weights * labels_sam, axis=1)
sess.run(weights_test)
sess.run(loss)
sess.run(mask)
weighted_losses = loss*weights_test



y_true = [1,0,0,0]
y_pred = [0,0,0,0]
cnf_matrix = confusion_matrix(y_true,y_pred)
precision_score(y_true, y_pred, average='macro')
'''