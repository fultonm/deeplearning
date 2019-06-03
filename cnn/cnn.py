from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten
from keras.utils.vis_utils import plot_model
from clean import clean_doc
from os import listdir

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def load_doc(filename):
    """
    Loads the document at the given filename
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text.split('\n')

# load and clean a dataset


def load_clean_dataset(is_train):
    # load documents
    neg = load_doc(
        'cleaned/neg_train.txt') if is_train else load_doc('cleaned/neg_test.txt')
    pos = load_doc(
        'cleaned/pos_train.txt') if is_train else load_doc('cleaned/pos_test.txt')
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels


def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def predict_sentiment(review, vocab, tokenizer, max_length, model):
    review = clean_doc(review)
    review = encode_docs(tokenizer, max_length, [review])
    yaht = model.predict(review, verbose = 0)
    percent_pos = yaht[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

def process_docs(directory):
    """
    For each document in the given directory,
    """
    docs = []
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        doc = '\n'.join(doc)
        docs.append([filename, doc])
    return docs

train_docs, train_labels = load_clean_dataset(is_train=True)

tokenizer = create_tokenizer(train_docs)
max_length = max([len(s.split()) for s in train_docs])
vocab_size = len(tokenizer.word_index) + 1

encoded_train_docs = encode_docs(tokenizer, max_length, train_docs)

model = None
try:
    model = load_model('model.h5')
    print('Loaded previously fitted model `model.h5` from disk')
except:
    print('Model doesn\'t exist; fitting model to movie data then saving to `model.h5` on disk')
    model = define_model(vocab_size, max_length)
    model.fit(encoded_train_docs, train_labels, epochs=10, verbose=2)
    model.save('model.h5')

_, acc = model.evaluate(encoded_train_docs, train_labels, verbose=0)
print('Train accuracy: %.2f' % (acc * 100))

test_docs, test_labels = load_clean_dataset(is_train=False)
encoded_test_docs = encode_docs(tokenizer, max_length, test_docs)

_, acc = model.evaluate(encoded_test_docs, test_labels, verbose=0)
print('Test accuracy: %.2f' % (acc * 100))

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab)

new_reviews = process_docs('new_reviews')
results = []

for new_review in new_reviews:
    result = predict_sentiment(new_review[1], vocab, tokenizer, max_length, model)
    results.append((new_review[0], result))

for result in results:
    print(result)
