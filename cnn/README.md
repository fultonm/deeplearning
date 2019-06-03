Run scripts in following order:

1) clean.py         # Reads from `txt_sentoken` directory, places cleaned reviews in the provided empty `cleaned` directory
2) build_vocab.py   # Creates a vocab file from the cleaned reviews and outputs it to vocab.txt
3) cnn.py   # Trains network, saves network to disk, and performs sentiment analysis using Convolutional neural network