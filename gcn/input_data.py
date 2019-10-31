import numpy as np
import tensorflow as tf
import random

flags = tf.app.flags
FLAGS = flags.FLAGS


def pollute_data_2(labels,attributes, idx_train, idx_val, idx_test):
    print ("Pollute data process starts \n")
    #transform attributes to csr format
    attributes = attributes.tocsr()
    #Get the max values for each column
    col_max = attributes.max(axis=0)
    col_max = col_max.toarray()
    attributes_array = attributes.toarray()

    #Get the max values' row indices for each column
    idx_all = range(len(labels))
    #num_train = round(len(idx_all)/(1+FLAGS.train_test_ratio))
    num_train = len(idx_train)
    num_test = len(idx_test)
    #num_test = len(idx_all)-num_train
    #idx_train = range(int(round(num_train*FLAGS.train_validation_ratio)))
    #idx_val = range(len(idx_train),int(num_train))
    #idx_test = range(int(num_train),int(len(idx_all)))

    #Get the shape of label, the size of columns of attributes should increase 1 to indicate clean data
    x = attributes.shape[0]
    y = attributes.shape[1]
    y += 1

    y_train = np.zeros((x,y))
    y_val =np.zeros((x,y))
    y_test = np.zeros((x,y))

    # pollute_train
    #Get the random_pollute_train_instance from training dataset
    pollute_train_size = int (len(idx_train)*FLAGS.pollute_ratio)
    pollute_train_row = np.random.choice(idx_train,pollute_train_size,replace=False)
    pollute_val_size = int(len(idx_val) * FLAGS.pollute_ratio)
    pollute_val_row = np.random.choice(idx_val, pollute_val_size, replace=False)
    pollute_test_size = int(len(idx_test) * FLAGS.pollute_ratio)
    pollute_test_row = np.random.choice(idx_test, pollute_test_size, replace=False)

    for i in idx_train:
        if i in pollute_train_row:
            col_pollute = np.random.randint(low=0,high=attributes.shape[1])
            #col_pollute = np.random.sample(range(attributes.shape[1]),round(attributes.shape[1]*FLAGS.attribute_pollution_ratio))
            #print (col_pollute)
            y_train[i][col_pollute] =1
            temp_max = col_max[0][col_pollute]
            attributes[i,col_pollute] = 2*temp_max
            print("pollute training data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_val_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_val[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute validation data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_test_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_test[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute testing data")
            print(i, col_pollute, 2*temp_max)
        else:
            y_train[i][attributes.shape[1]]=1
            print(i)

    #pollute_validation
    for i in idx_val:
        if i in pollute_train_row:
            col_pollute = np.random.randint(low=0,high=attributes.shape[1])
            y_train[i][col_pollute] =1
            temp_max = col_max[0][col_pollute]
            attributes[i,col_pollute] = 2*temp_max
            print("pollute training data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_val_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_val[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute validation data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_test_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_test[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute testing data")
            print(i, col_pollute, 2*temp_max)
        else:
            y_train[i][attributes.shape[1]]=1
            print(i)


    #pollute testing
    for i in idx_test:
        if i in pollute_train_row:
            col_pollute = np.random.randint(low=0,high=attributes.shape[1])
            y_train[i][col_pollute] =1
            temp_max = col_max[0][col_pollute]
            attributes[i,col_pollute] = 2*temp_max
            print("pollute training data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_val_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_val[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute validation data")
            print(i, col_pollute, 2*temp_max)
        elif i in pollute_test_row:
            col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            y_test[i][col_pollute] = 1
            temp_max = col_max[0][col_pollute]
            attributes[i, col_pollute] = 2 * temp_max
            print("pollute testing data")
            print(i, col_pollute, 2*temp_max)
        else:
            y_train[i][attributes.shape[1]]=1
            print(i)

    #print ("test")
    labels =  np.zeros((x,y))
    labels = y_train +y_val + y_test
    #print labels.shape
    return attributes, labels

def pollute_data(labels,attributes, idx_train, idx_val, idx_test):
    print ("Pollute data process starts \n")
    # transform attributes to csr format
    attributes = attributes.tocsr()
    #Get the shape of label
    x = attributes.shape[0]
    #We map each attribute pollution bit as one label, and add additional 1 label for clean
    #y=attributes.shape[1]+1
    y=2
    y_train = np.zeros((x, y))
    y_val = np.zeros((x, y))
    y_test = np.zeros((x, y))

    # pollute_train
    # Get the random_pollute_train_instance from training dataset
    pollute_train_size = int(len(idx_train) * FLAGS.pollute_ratio)
    pollute_train_row = np.random.choice(idx_train, pollute_train_size, replace=False)
    pollute_val_size = int(len(idx_val) * FLAGS.pollute_ratio)
    pollute_val_row = np.random.choice(idx_val, pollute_val_size, replace=False)
    pollute_test_size = int(len(idx_test) * FLAGS.pollute_ratio)
    pollute_test_row = np.random.choice(idx_test, pollute_test_size, replace=False)

    temp_count =0

    for i in idx_train:
        if i in pollute_train_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            col_pollute = random.sample(range(attributes.shape[1]),int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio),1)
            print (col_pollute)
            #Always use the second bit to indicate dirty instance
            y_train[i][1] = 1
            #y_train[i][col_pollute]= 1

            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                print("pollute training data")
                print(i, j, attributes[i,j], temp_count)
            '''
            if attributes[i,col_pollute]==0:
                attributes[i, col_pollute] = 1
            else:
                attributes[i, col_pollute] = 0
            '''
            temp_count+=1
        else:
            #y_train[i][y-1] = 1
            y_train[i][0]=1
            #print(i)
    for i in idx_val:
        if i in pollute_val_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            col_pollute = random.sample(range(attributes.shape[1]),
                                        int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio), 1)
            print (col_pollute)
            y_val[i][1] = 1
            #y_val[i][col_pollute] = 1

            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                #print("pollute validation data")
                #print(i, j, attributes[i, j], temp_count)
            '''
            if attributes[i, col_pollute] == 0:
                attributes[i, col_pollute] = 1
            else:
                attributes[i, col_pollute] = 0
            '''
            temp_count += 1
        else:
            y_val[i][0]= 1
            #y_val[i][y-1]=1
            #print(i)
    for i in idx_test:
        if i in pollute_test_row:
            #col_pollute = np.random.randint(low=0, high=attributes.shape[1])
            col_pollute = random.sample(range(attributes.shape[1]),
                                        int(attributes.shape[1] * FLAGS.attribute_pollution_ratio))
            col_pollute = np.array(col_pollute)
            col_pollute.reshape(int(attributes.shape[1] * FLAGS.attribute_pollution_ratio), 1)
            print (col_pollute)
            y_test[i][1] = 1
            #y_test[i][col_pollute]=1

            for j in col_pollute:
                if attributes[i,j]==0:
                    attributes[i,j]=1
                else:
                    attributes[i,j]=0
                #print("pollute testing data")
                #print(i, j, attributes[i, j], temp_count)
            '''
            if attributes[i, col_pollute] == 0:
                attributes[i, col_pollute] = 1
            else:
                attributes[i, col_pollute] = 0
            '''
            temp_count += 1
        else:
            y_test[i][0]= 1
            #y_test[i][y-1]=1
            #print(i)

    labels = np.zeros((x, y))
    labels = y_train + y_val + y_test
    # print labels.shape
    return attributes, labels
