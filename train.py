# train.py

import pandas as pd
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# first read every single csv files and make one csv file for training only - make function!!!

def seperate_data(path, num, span):
  # make new csv file
  # 
  # 

def load_data(span, group):
  
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
