import matplotlib.pyplot as plt
from keras.datasets import reuters
import numpy as np
from keras import layers
from keras import models
import matplotlib
matplotlib.use("TkAgg")

words = 10000
word_index = reuters.get_word_index()
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])


def get_text(integers):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in integers])


def vectorize_sequence(sequences, dimension=words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # actually another iterator of sorts,
        results[i, sequence] = 1
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=words)

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(words,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(46, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = network.fit(partial_x_train, partial_y_train,
                      epochs=9, batch_size=512, validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)
