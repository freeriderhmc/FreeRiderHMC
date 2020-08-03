import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np



def dataProcess(tracknum, cnt, span, lanechng=(0,0,0), turn=(0,0,0)):
    tmpdf = pd.read_csv('{}.csv'.format(tracknum), index_col = 0)
    tmpdf["ans"]=0
    if(lanechng[0]==1):
        tmpdf.at[lanechng[1]:lanechng[2], 'ans'] = 1
    elif(turn[0]==1):
        tmpdf.at[turn[1]:turn_fin[2], 'ans'] = 2
    tmpdf = tmpdf.dropna(axis=0)
    data = []
    y_data = []
    for i in range(0, len(tmpdf)-cnt):
        tmplist = []
        for j in range(i, cnt):
            tmplist.append(j)
        print(tmpdf.iloc[i+cnt+span-1,5])
        if(tmpdf.iloc[i+cnt+span-1,5]==0):
            data.append(tmpdf.iloc[tmplist, [0,1,3]].to_numpy())
            y_data.append(0)
        elif(tmpdf.iloc[i+cnt+span-1,5]==1):
            data.append(tmpdf.iloc[tmplist, [0,1,3]].to_numpy())
            y_data.append(1)
        elif(tmpdf.iloc[i+cnt+span-1,5]==2):
            data.append(tmpdf.iloc[tmplist, [0,1,3]].to_numpy())
            y_data.append(2)
        else:
            continue
    n_train = int(len(data)*0.8)
    n_test = int(len(data) - n_train)
    
    X_test = data[n_train:]
    y_test = np.array(y_data[n_train:])
    X_train = data[:n_train]
    y_train = np.array(y_data[:n_train])
    
    return X_train, y_train, X_test, y_test

def LSTM_RNN(_X, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # LSTM cells
    lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    # many-to-one
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_, n_classes=n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
