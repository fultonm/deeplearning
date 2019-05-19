from keras.datasets import imdb
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

words = 10000

# Import the data using a dictionary with 10000 words
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=words)

# Print the data to see its shape
# print(train_data[0])
# print(train_labels)

# The maximum number representing words is no greater than our num_words above
#print(max([max(w) for w in train_data]))

# We can map the integers representing words to the words the acutally represent,
# creating a paragraph that is mostly human readable.
word_index = imdb.get_word_index()
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?')
                           for i in train_data[0]])
# print(decoded_review)

# We need to vectorize the data so we can feed it into the network.


def vectorize_sequences(sequences, dimension=words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(words,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = network.fit(partial_x_train, partial_y_train,
                      epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

