from sklearn.feature_extraction.text import TfidfVectorizer

# List of text documents ? ( Looks like we have one documents )
text = ['The quick brown fox jumped over the lazy dog.', 'The dog.', 'The fox']
# Create the transform instance
vectorizer = TfidfVectorizer()
# Tokens and build the vocab
vectorizer.fit(text)
# Summarize
print('vocab:')
print(vectorizer.vocabulary_)
print('idf:')
print(vectorizer.idf_)

# Encode the document
vector = vectorizer.transform([text[0]])
# Summarize encoded vector
print('shape:')
print(vector.shape)
print('type:')
print(type(vector))
print('vector:')
print(vector)