from os import listdir
from clean import clean_doc
from bow import load_doc, vocab_filename, train_model

vocab = set(load_doc(vocab_filename))
model, tokenizer = train_model()


def predict_sentiment(doc, vocab, tokenizer, model):
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    y_hat = model.predict(encoded, verbose=0)
    percent_pos = y_hat[0, 0]
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

docs = process_docs('new_reviews')
results = []

for filename_and_doc in docs:
    results.append([filename_and_doc[0], predict_sentiment(filename_and_doc[1], vocab, tokenizer, model)])

for result in results:
    print(result)