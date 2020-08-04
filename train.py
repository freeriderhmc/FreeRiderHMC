# train.py

import pandas as pd
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

cnt = 10 #
span = 8 #

# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 3
learning_rate = 0.01 #
lambda_loss_amount = 0.0015 #

batch_size = 8 
display_iter = 1000  # show test set accuracy during training

def scailing(dataframe):
    result_x = dataframe['0']
    result_y = dataframe['1']
    x_mean = result_x.mean()
    x_std = result_x.std()
    y_mean = result_y.mean()
    y_std = result_y.std()
    result_x = (result_x-x_mean)/x_std
    result_y = (result_y-y_mean)/y_std
    res = pd.DataFrame({'x':result_x, 'y':result_y, 'yaw':dataframe['3'], 'ans':dataframe['ans']})
    return res

def dataAssembly(tracknum, lanechng=(0,0,0), turn=(0,0,0)):
    tmpdf = pd.read_csv('/content/gdrive/My Drive/{}.csv'.format(tracknum), index_col = 0)
    tmpdf["ans"]=0
    if(lanechng[0]==1):
        tmpdf.at[lanechng[1]:lanechng[2], 'ans'] = 1
    elif(turn[0]==1):
        tmpdf.at[turn[1]:turn[2], 'ans'] = 2
    tmpdf = tmpdf.dropna(axis=0)
    tmpdf = scailing(tmpdf)
    return tmpdf


def dataProcess(tmpdf, cnt, span):
    data = []
    y_data = []
    for i in range(0, len(tmpdf)-cnt-span):
        tmplist = []
        for j in range(i, cnt+i):
            tmplist.append(j)
        if(tmpdf.iloc[i+cnt+span,3]==0):
            data.append(tmpdf.iloc[tmplist, [0,1,2]].to_numpy())
            y_data.append(0)
        elif(tmpdf.iloc[i+cnt+span,3]==1):
            data.append(tmpdf.iloc[tmplist, [0,1,2]].to_numpy())
            y_data.append(1)
        elif(tmpdf.iloc[i+cnt+span,3]==2):
            print('here')
            data.append(tmpdf.iloc[tmplist, [0,1,2]].to_numpy())
            y_data.append(2)
        else:
            continue
    n_train = int(len(data)*0.8)
    n_test = int(len(data) - n_train)
    
    X_test = np.array(data[n_train:])
    y_test = np.array(y_data[n_train:])
    X_train = np.array(data[:n_train])
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

LABELS = [
    "NORMAL", 
    "LANE_CHANGE", 
    "TURN"
]


X_train = np.empty([0,cnt,3])
X_test = np.empty([0,cnt,3])
y_train = np.empty([0,])
y_test = np.empty([0,])
tmp_dataframe = pd.DataFrame()

label_df = pd.read_csv('/content/gdrive/My Drive/labeling.csv', index_col = 0)
for index, row in label_df.iterrows():
    tracknum = index
    if(np.isnan(row['start_lnchn'])):
        lanechng = (0,0,0)
        turn = (1, row['start_turn'], row['fin_turn'])
        print(lanechng, turn)
    elif(np.isnan(row['start_turn'])):
        lanechng = (1, row['start_lnchn'], row['fin_lnchn'])
        turn = (0,0,0)
        print(lanechng, turn)
    tmp_dataframe = pd.concat([tmp_dataframe,dataAssembly(tracknum, lanechng, turn)])
X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = dataProcess(tmp_dataframe, cnt, span) #track num
X_train = np.append(X_train, X_train_tmp, axis=0)
y_train = np.append(y_train, y_train_tmp, axis = 0)
X_test = np.append(X_test, X_test_tmp, axis=0)
y_test = np.append(y_test,  y_test_tmp, axis = 0)

    
training_data_count = len(X_train)
test_data_count = len(X_test)
n_steps = len(X_train[0]) #number of timestamp
n_input = len(X_train[0][0])  # how many input parameters per timestamp

training_iters = training_data_count * 300  # Loop 300 times on the dataset

# shape, normalization
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

print("number of training data : ", training_data_count)
print("number of test data : ", test_data_count)
print("number of cound hard coding : ", cnt, "number of steps : ", n_steps)
print("number of input : ", n_input)

'''
epoch = 100
batchsize = 64
model(X_train, y_train, cnt, epoch, batchsize, X_test, y_test)
'''

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Done")


# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()

# Results
predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
