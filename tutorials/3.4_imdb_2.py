from keras.datasets import imdb
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

words = 10000

# Import the data using a dictionary with 10000 words
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=words)

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
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy', metrics=['accuracy'])

history = network.fit(x_train, y_train,
                      epochs=4, batch_size=512)

results = network.evaluate(x_test, y_test)

print(results)