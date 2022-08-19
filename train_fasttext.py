
import fasttext

# Skipgram model :
model = fasttext.train_unsupervised('data.txt', model='skipgram')