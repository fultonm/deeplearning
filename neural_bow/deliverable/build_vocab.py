from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import string
import re

pos_train_filename = 'cleaned/pos_train.txt'
neg_train_filename = 'cleaned/neg_train.txt'
vocab_filename = 'vocab.txt'


def load_doc(filename):
    """
    Loads the document at the given filename
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text.split('\n')


def save_vocab(lines, filename):
    """
    Saves vocabulary lines as newline-separated values at the given filename
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def add_docs_to_vocab(docs, vocab):
    """
    For each document in the given directory,
    Add them to the vocab Counter
    """
    for doc in docs:
        tokens = doc.split()
        vocab.update(tokens)

pos_train = load_doc(pos_train_filename)
neg_train = load_doc(neg_train_filename)

vocab = Counter()
add_docs_to_vocab(pos_train, vocab)
add_docs_to_vocab(neg_train, vocab)

print(len(vocab))

""" Remove all tokens which occur less than 2 times """
min_occurrence = 2
tokens = [t for t, c in vocab.items() if c >= min_occurrence]

print(len(tokens))

""" 
Saving our processed vocab to a file allows us to decouple 
the rest of the process from vocab generation step
"""
save_vocab(tokens, vocab_filename)