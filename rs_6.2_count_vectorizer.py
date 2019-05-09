from sklearn.feature_extraction.text import CountVectorizer

# List of text documents ? ( Looks like we have one documents )
text = ['The quick brown fox jumped over the lazy dog.']
# Create the transform instance
vectorizer = CountVectorizer()
# Tokens and build the vocab
vectorizer.fit(text)
# Summarize
print(vectorizer.vocabulary_)

# Encode the document
vector = vectorizer.transform(text)
# Summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector)