

model = Sequential()
# model.add(Embedding(3, 32)) # embedding vector 32 levels
model.add(LSTM(256, input_shape=(seq_len, 3))) # RNN cell hidden_size 32, SimpleRNN
model.add(Dense(3, activation='softmax')) #if classify->sigmoid

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#optimizer rmsprop
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
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
