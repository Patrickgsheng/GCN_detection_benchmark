import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

from sklearn import metrics



def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)

    '''
    # Added for weighted classfication
    mask_1 =tf.broadcast_to(mask,[2,FLAGS.total_instance])
    mask_1 =tf.transpose(mask_1)
    labels_sum = tf.reduce_sum(labels*mask_1,0)
    x_max = tf.reduce_max(labels_sum)
    class_weights = x_max / labels_sum
    weights_test = tf.reduce_sum(class_weights * labels, axis=1)
    #########################
    '''

    mask /= tf.reduce_mean(mask)
    loss *= mask

    #modify the loss to weighted loss
    #loss *= weights_test
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_precision(preds, labels, mask):
    y_true = tf.argmax(labels, 1)
    #print("The ground truth is %s" %str(y_true.shape))
    y_pred = tf.argmax(preds, 1)
    #print("The prediction is %d" % (y_pred))
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    #with tf.Session() as sess: print(y_true)
    #with tf.Session() as sess: print(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    #with tf.Session() as sess: print(TP)
    #with tf.Session() as sess: print(FP)
    #print("Number of TP is %d"%(TP))
    #print("Number of FP is %d"%(FP))
    precision = tf.divide(TP, TP + FP)
    return precision


'''
def masked_precision(preds, labels, mask):
    y_true = tf.argmax(labels, 1)
    # print("The ground truth is %s" %str(y_true.shape))
    y_pred = tf.argmax(preds, 1)
    # print("The prediction is %d" % (y_pred))
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    precision = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    return precision[-1]
'''

def masked_recall(preds, labels, mask):
    '''Recall with masking. '''
    y_true = tf.argmax(labels, 1)
    y_pred = tf.argmax(preds, 1)
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    recall = tf.divide(TP, TP + FN)
    return recall

def masked_f1(preds, labels, mask):
    '''Recall with masking. '''
    y_true = tf.argmax(labels, 1)
    y_pred = tf.argmax(preds, 1)
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    precision = tf.divide(TP, TP + FP)
    recall = tf.divide(TP, TP + FN)
    f1_score = 2*precision*recall /(precision+recall)
    return f1_score

def masked_TP(preds, labels,mask):
    y_true = tf.argmax(labels, 1)
    y_pred = tf.argmax(preds, 1)
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    TP = tf.count_nonzero(y_pred * y_true)


    return TP

def masked_FP(preds, labels,mask):
    y_true = tf.argmax(labels, 1)
    y_pred = tf.argmax(preds, 1)
    mask.set_shape([None])
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    FP = tf.count_nonzero(y_pred * (y_true - 1))

    return FP