
from keras.preprocessing.text import Tokenizer

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


vocab = set(load_doc(vocab_filename))

train_docs, y_train = load_dataset(
    vocab, pos=pos_train_filename, neg=neg_train_filename)
test_docs, y_test = load_dataset(
    vocab, pos=pos_test_filename, neg=neg_test_filename)

tokenizer = create_tokenizer(train_docs)

x_train = tokenizer.texts_to_matrix(train_docs, mode='freq')
x_test = tokenizer.texts_to_matrix(test_docs, mode='freq')

print(x_train.shape, x_test.shape)
