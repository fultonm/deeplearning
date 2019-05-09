import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from ntlk.stem.porter import PorterStemmer

stop_words = stopwords.words('english')
# Regex for filtering out punctuation
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# Regex for filtering out non-printable characters
re_print = re.compile('[^%s]' % re.escape(string.printable))

# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# Tokenize the input text into an array of words
tokens = word_tokenize(text)
# Convert all words to lowercase
tokens = [t.lower() for t in tokens]
# Filter out punctuation
tokens = [re_punc.sub('', t) for t in tokens]
# Filter non-printables
tokens = [re_print.sub('', t) for t in tokens]
# Filter non-alphabetic words
tokens = [w for w in tokens if w.isalpha()]
# Filter out stop words
tokens = [t for t in tokens if not t in stop_words]

print(tokens[:100])
