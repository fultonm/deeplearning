Run scripts in following order:

1) clean.py         # Reads from `txt_sentoken` directory, places cleaned reviews in the provided empty `cleaned` directory
2) build_vocab.py   # Creates a vocab file from the cleaned reviews and outputs it to vocab.txt
3) new_reviews.py   # Trains network and gives sentiment scores to all reviews contained in `new_reviews.py`, outputs to stdout