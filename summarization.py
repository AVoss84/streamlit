import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
import networkx as nx
import os
from sklearn.feature_extraction.text import CountVectorizer

from importlib import reload
from utils import utils as util

reload(util)

#nltk.download('stopwords')

# https://github.com/crabcamp/lexrank

#https://github.com/edubey/text-summarizer

# https://jdvala.github.io/blog.io/thesis/2018/05/11/German-Preprocessing.html

stopWords = set(stopwords.words('german'))

print(stopWords)


corpus, tokens = util.read_article('data/raw_text.txt')
corpus

util.generate_summary('data/fb.txt', 2, 'english')
util.generate_summary('data/raw_text.txt', 1, 'german')
#util.generate_summary('data/german.txt', 1, 'german')


words = word_tokenize(corpus)
wordsFiltered = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)

print(wordsFiltered)



len(corpus)


for i in corpus: 
    print(i)

vec = CountVectorizer(lowercase=True, #ngram_range=(2, 2),
                token_pattern = '(?u)(?:(?!\d)\w)+\\w+', 
                analyzer = 'word',  #char_wb
                tokenizer = None, 
                stop_words = stopWords #"english       
                ) 

vec.stop_words

vec.fit_transform()