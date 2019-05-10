from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
import re
import numpy
import sys

numpy.set_printoptions(threshold=sys.maxsize)

def load_doc(filename):  
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def process_docs(directory):
    docs = []
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        doc = load_doc('{}/{}'.format(directory, filename))
        docs.append(doc)
    return docs


def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

directory = 'txt_sentoken/neg'
docs = process_docs(directory)
docs = [clean_doc(doc) for doc in docs]

print(len(docs))
vectorizer = CountVectorizer()
bag = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names())
print(bag.toarray()[0])


#print(encoded_doc)