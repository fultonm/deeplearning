from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import string
import re

neg_dir = 'txt_sentoken/neg'
pos_dir = 'txt_sentoken/pos'
vocab_filename = 'vocab.txt'
clean_pos_review_filename = 'positive.txt'
clean_neg_review_filename = 'negative.txt'


def load_doc(filename):
    """
    Loads the document at the given filename
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def save_vocab(lines, filename):
    """
    Saves vocabulary lines as newline-separated values at the given filename
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def clean_doc(doc):
    """
    Cleans the given doc by
    1) Removing punctuation as defined by string.punctuation
    2) Filters out any non-alpha words as defined by isalpha()
    3) Removes all stopwords as defined by nltk's stopwords list
    4) Removes all 1-character words
    """
    tokens = doc.split()
    # Remove punctuation marks [., ;, :, ", e.t.c]
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    # Keep only alphabetic words (Remove words with numbers)
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words like "and, but, a, the, an" e.t.c
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # Remove all other 1-character words if they're still in there
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    """
    Adds all new tokens to the vocab Counter which are present in the doc
    at the given filename
    """
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def process_docs(directory, vocab):
    """
    For each document in the given directory,
    Add them to the vocab Counter
    """
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)


vocab = Counter()
process_docs(neg_dir, vocab)
process_docs(pos_dir, vocab)
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


def doc_to_line(filename, vocab):
    """
    Returns a single line containing the cleaned review
    """
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [t for t in tokens if t in vocab]
    return ' '.join(tokens)


def process_docs_2(directory, vocab):
    """
    For each document in the directory,
    Generate a single line with the cleaned review document. 
    Return all the documents as a list of cleaned lines
    """
    lines = []
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


def save_review_to_file(lines, filename):
    """
    Saves a collection of reviews to file,
    one review per line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

negative_lines = process_docs_2(neg_dir, vocab)
positive_lines = process_docs_2(pos_dir, vocab)

save_review_to_file(negative_lines, clean_neg_review_filename)
save_review_to_file(positive_lines, clean_pos_review_filename)
