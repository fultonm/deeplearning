from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick

#documet
text = 'The quick brown fox jumped over the lazy dog.'
words = set(text_to_word_sequence(text))
vocab_size = len(words)

print(vocab_size)

result = hashing_trick(text, round(vocab_size * 1.33), hash_function='md5')
print(result)