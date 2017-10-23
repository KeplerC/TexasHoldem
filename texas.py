
# coding: utf-8

# In[1]:

SCORE_FOLD = 1 #if expected score is below two pairs, then fold

#import libraries 
import pandas as pd
import tempfile
import urllib



# In[2]:

#read training file from UCI dataset 
#train_file = tempfile.NamedTemporaryFile()
#urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data", train_file.name)

CSV_COLUMNS = ["S1","C1","S2","C2","S3","C3","S4","C4", "S5","C5","Hand"]
read_df = pd.read_csv("./poker-hand-training-true.data", names=CSV_COLUMNS, skipinitialspace=True)
#read_df = pd.read_csv("./poker-hand-training.data", names=CSV_COLUMNS, skipinitialspace=True)
print(read_df)


# In[3]:

#generate training cases 
#because we have only two cards available, we generate two combinations with corresponding result 

import itertools

#select two combinations 
train_list = list(itertools.combinations(range(0,10,2), 2))
print(train_list)

frames = []
for target in train_list:
    local_train_list = [target[0], target[0]+1, target[1], target[1]+1, 10] #append the final result to our table 
    local_data_frame = read_df.iloc[:, local_train_list]
    local_data_frame.columns = ["S1","C1","S2","C2", "Hand"]
    local_data_frame['diff_S'] = abs(local_data_frame['S1'] - local_data_frame['S2'])
    local_data_frame['diff_C'] = abs(local_data_frame['C1'] - local_data_frame['C2'])
    local_data_frame['diff_S_sc'] = abs((local_data_frame['S1'] - local_data_frame['S2'])) * 3
    local_data_frame['sum_C'] = local_data_frame['C1'] + local_data_frame['C2']
    local_data_frame['prod_C'] = (local_data_frame['C1'] * local_data_frame['C2']) / 10
    frames.append(local_data_frame)
training_df = pd.concat(frames)
print(training_df)


# In[4]:

#generate label
#do this based on SCORE-FOlD: how expected hand you will fold
train_labels = (training_df["Hand"].apply(lambda x: x > SCORE_FOLD)).astype(int)
training_df = training_df.drop('Hand', 1)
print(train_labels)
print(training_df)


# In[5]:

import numpy as np

print(training_df. shape)
trainX = training_df.iloc[:, 0:9].values
trainY = train_labels.values

learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 50
m, n = trainX.shape
#m: number of training samples 
#n: number of features

print(trainX.shape)
print(trainY.shape)


def fetch_batch(X_train, Y_train, batch_index, batch_size):
    X_batch = X_train[batch_index*batch_size: (batch_index+1)*batch_size, :]
    y_batch = Y_train[batch_index*batch_size: (batch_index+1)*batch_size]
    return X_batch, y_batch
    


# In[ ]:

#tensorflow
#do logistic regression 
import tensorflow as tf
import numpy as np
import sys 

#enable GPU acceleration

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
#shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

    
with tf.device(device_name):

    #graph input
    x = tf.placeholder(tf.float32, [None, n])
    Y = tf.placeholder(tf.float32)

    #weights 
    #W = tf.Variable(tf.zeros([n, 1]), name="weight")
    W = tf.random_uniform([n, 1], -1.0, 1.0)
    b = tf.Variable(np.random.randn(), name="bias")

    #model
    pred = tf.nn.sigmoid(tf.matmul(x, W) + b)

    #cost 
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))

    #optimizer is Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #saver
    saver = tf.train.Saver()


#training
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #saver.restore(sess, "./model.ckpt")

    for epoch in range(training_epochs):
        average_cost = 0
        total_batch = int(m / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = fetch_batch(trainX, trainY, i, batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, Y: batch_y})
            average_cost += c/total_batch
            #print(W.eval())
        
        if(epoch+1) % display_step == 0:
            print("Epoch:", '%05d' % (epoch+1), "cost=", "{:.20f}".format(average_cost))
            print("Now the weight is :" + str(W.eval()) + "\n")
            print("Now the weight is :" + str(b.eval()) + "\n")
            save_path = saver.save(sess, "./model.ckpt")
            
    print("Optimization Finished!")
    
    #model evaluation
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: trainX, Y: trainY}))


# In[9]:

#evaluation 

