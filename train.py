# train.py

import pandas as pd
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# first read every single csv files and make one csv file for training only - make function!!!

# 1.csv to (1_normal.csv and 1_lanechng.csv 1_left.csv)
def seperate_data(tracknum):
    rawdf = pd.read_csv('{}.csv'.format(tracknum), index_col=0)
    ansdf = pd.read_csv('{}_ans.csv'.format(tracknum), index_col = 0)
    temp = pd.merge(rawdf, ansdf, left_on="0", left_index = True,right_index = True, how='left').dropna(axis=0)
    normal = temp[temp['ans']==0]
    lanechng = temp[temp['ans']==1]
    left = temp[temp['ans']==2]
    pd.DataFrame(normal).to_csv('{}_normal.csv'.format(tracknum))
    pd.DataFrame(lanechng).to_csv('{}_lanechng.csv'.format(tracknum))
    pd.DataFrame(left).to_csv('{}_left.csv'.format(tracknum))

# in one data folder
def load_data(cnt, span, group):
  
''' 
def sequential_window_dataset(series, window_size):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda window: (window[:-1], window[1:]))
    return ds.batch(1).prefetch(1)
'''

# window_size = #

dataset = pd.read_csv('training.csv', index_col=0)

# Extract out states and labels
states = dataset['vecs'].tolist()
labels = dataset['state'].tolist()

n_train = int(len(states) * 0.8)
n_test = int(len(states) - n_train)
print('num of train datasets :',n_train)
print('num of test datasets:',n_test)

X_test = data[n_train:]
y_test = np.array(y_data[n_train:])
X_train = data[:n_train]
y_train = np.array(y_data[:n_train])

model = Sequential()
# model.add(Embedding(3, 32)) # embedding vector 32 levels
model.add(LSTM(256, input_shape=(seq_len, 3))) # RNN cell hidden_size 32, SimpleRNN
model.add(Dense(2, activation='softmax')) #if classify->sigmoid

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#optimizer rmsprop
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n test accuracy: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
