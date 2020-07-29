# train.py

import pandas as pd
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# first read every single csv files and make one csv file for training only - make function!!!

def seperate_data(tracknum, cnt, span):
    rawdf = pd.read_csv('{}.csv'.format(tracknum), index_col=0)
    ansdf = pd.read_csv('{}_ans.csv'.format(tracknum), index_col = 0)
    temp = pd.merge(rawdf, ansdf, left_on="0", left_index = True,right_index = True, how='left').dropna(axis=0)
    data = []
    y_data = []
    for i in range(0, len(temp)-cnt):
        tmplist = []
        for j in range(i, cnt):
            tmplist.append(j)
        if(temp.iloc[i+cnt+span,5]==0):
            data.append(temp.iloc[tmplist,[0,1,3]].to_numpy())
            y_data.append(0)
        elif(temp.iloc[i+cnt+span,5]==1):
            data.append(temp.iloc[tmplist,[0,1,3]].to_numpy())
            y_data.append(1)
        elif(temp.iloc[i+cnt+span,5]==2):
            data.append(temp.iloc[tmplist,[0,1,3]].to_numpy())
            y_data.append(2)
        else:
            continue
    n_train = int(len(data) * 0.8)
    n_test = int(len(data) - n_train)
    
    X_test = data[n_train:]
    y_test = np.array(y_data[n_train:])
    X_train = data[:n_train]
    y_train = np.array(y_data[:n_train])
    
    return X_train, y_train, X_test, y_test

# window_size = #
'''
dataset = pd.read_csv('training.csv', index_col=0)

# Extract out states and labels
states = dataset['vecs'].tolist()
labels = dataset['state'].tolist()
'''

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
