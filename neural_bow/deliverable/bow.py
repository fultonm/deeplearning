
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
from keras.models import Sequential
from pandas import DataFrame
from matplotlib import pyplot

pos_train_filename = 'cleaned/pos_train.txt'
neg_train_filename = 'cleaned/neg_train.txt'

pos_test_filename = 'cleaned/pos_test.txt'
neg_test_filename = 'cleaned/neg_test.txt'

vocab_filename = 'vocab.txt'


def load_doc(filename):
    """
    Loads the document at the given filename
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text.split('\n')


def filter_doc_vocab(doc, vocab):
    """
    Filters all words in the doc not present in the vocab
    """
    tokens = doc.split()
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def load_dataset(vocab, pos, neg):
    """
    Returns two lists: a concatentated negative and postive reviews list, 
    and a coorepsonding list of labels denoting positive or negative.
    """
    pos_docs = load_doc(pos)
    pos_docs = [filter_doc_vocab(d, vocab) for d in pos_docs]

    neg_docs = load_doc(neg)
    neg_docs = [filter_doc_vocab(d, vocab) for d in neg_docs]

    docs = pos_docs + neg_docs
    labels = [1 for _ in range(len(pos_docs))] + \
        [0 for _ in range(len(neg_docs))]

    return docs, labels


def create_tokenizer(docs):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    return tokenizer

def define_model(n_words):
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words, ), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True)
    return model

def prepare_data(train_docs, test_docs, mode):
    tokenizer = create_tokenizer(train_docs)
    x_train = tokenizer.texts_to_matrix(train_docs, mode=mode)
    x_test = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return x_train, x_test, tokenizer

def evaluate_mode(x_train, y_train, x_test, y_test):
    scores = []
    n_repeats = 10
    n_words = x_test.shape[1]
    for i in range(n_repeats):
        model = define_model(n_words)
        model.fit(x_train, y_train, epochs=10, verbose=2)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i+1), acc))
    return scores

def train_model():
    vocab = set(load_doc(vocab_filename))
    train_docs, y_train = load_dataset(
        vocab, pos=pos_train_filename, neg=neg_train_filename)
    test_docs, y_test = load_dataset(
        vocab, pos=pos_test_filename, neg=neg_test_filename)
    x_train, x_test, tokenizer = prepare_data(train_docs, test_docs, mode='binary')
    n_words = x_train.shape[1]
    model = define_model(n_words)
    model.fit(x_train, y_train, epochs=10, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, tokenizer






# modes = ['binary', 'count', 'tfidf', 'freq']
# results = DataFrame()

# for mode in modes:
#     x_train, x_test = prepare_data(train_docs, test_docs, mode=mode)
#     results[mode] = evaluate_mode(x_train, y_train, x_test, y_test)
