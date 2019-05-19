from os import listdir
from nltk.corpus import stopwords
import string
import re

neg_dir = 'txt_sentoken/neg'
pos_dir = 'txt_sentoken/pos'
vocab_filename = 'vocab.txt'
clean_pos_review_filename = 'positive.txt'
clean_neg_review_filename = 'negative.txt'


def save(lines, filename):
    """
    Saves vocabulary lines as newline-separated values at the given filename
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_doc(filename):
    """
    Loads the document at the given filename
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


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


def process_docs(directory, start, end):
    """
    For each document in the given directory,
    """
    docs = []
    for filename in listdir(directory):
        number = int(filename[2:5])
        if not filename.endswith('.txt') or number < start or number > end:
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        cleaned_doc = clean_doc(doc)
        docs.append(' '.join(cleaned_doc))
    return docs


pos_train = process_docs(pos_dir, start=0, end=899)
pos_test = process_docs(pos_dir, start=900, end=999)
neg_train = process_docs(neg_dir, start=0, end=899)
neg_test = process_docs(neg_dir, start=900, end=999)

save(pos_train, 'cleaned/pos_train.txt')
save(pos_test, 'cleaned/pos_test.txt')
save(neg_train, 'cleaned/neg_train.txt')
save(neg_test, 'cleaned/neg_test.txt')
