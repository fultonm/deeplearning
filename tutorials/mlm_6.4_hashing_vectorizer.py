from sklearn.feature_extraction.text import HashingVectorizer

# List of text documents ? ( Looks like we have one documents )
text = ['The quick brown fox jumped over the lazy dog.']
# Create the transform instance
vectorizer = HashingVectorizer(n_features=20)
# Encode the document
vector = vectorizer.transform(text)
# Summarize encoded vector
print('shape:')
print(vector.shape)
print('type:')
print(type(vector))
print('vector:')
print(vector.toarray())
