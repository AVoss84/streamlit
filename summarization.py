
from copy import deepcopy
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
import pandas as pd
import networkx as nx
import os
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from importlib import reload

from utils import utils as util
from services import file
from pathlib import Path

#Path.home()
# docs = Path.cwd() / 'data/'
# file = docs / 'new_file.txt'
# file.touch()    # new file

# docs = Path.cwd() / 'data/'
# files = docs / 'new_file.txt'
# files

# os.path.join(docs, 'new_file.txt')

#nltk.download('stopwords')

# https://github.com/crabcamp/lexrank

#https://github.com/edubey/text-summarizer

# https://jdvala.github.io/blog.io/thesis/2018/05/11/German-Preprocessing.html


reload(util)
reload(file)

# Read corpus:
pre = util.text_tools()
corpus = pre.read_article(path = Path.cwd() / 'data/raw_text.txt')    # german
#corpus = pre.read_article(path = Path.cwd() / 'data/german.txt')    # german
#corpus = pre.read_article(path = Path.cwd() / 'data/fb.txt')    # english
corpus

# Clean corpus:
cleaner = util.clean_text(language = 'german')
corpus_train = cleaner.fit_transform(corpus)['text'].tolist()
corpus_train


#myyaml = file.YAMLservice(child_path = "config")

#repl_dict = myyaml.doRead(filename = "preproc_txt.yaml")

#dic = repl_dict['replace']['german']['umlaute']

# tt = util.text_tools()

# docs = tt.iter_document(corpus_cl.values.tolist())

# next(docs)

#df2 = df.replace({"text": dic})

#sent = pre.iter_document(corpus)
#next(sent)

#corpus, tokens = util.read_article('data/raw_text.txt')

#util.generate_summary('data/fb.txt', 2, 'english')
#util.generate_summary('data/raw_text.txt', 1, 'german')
#util.generate_summary('data/german.txt', 1, 'german')

# vec = CountVectorizer(lowercase=True, #ngram_range=(2, 2),
#                 token_pattern = '(?u)(?:(?!\d)\w)+\\w+', 
#                 analyzer = 'word',  #char_wb
#                 tokenizer = None, 
#                 stop_words = stopWords #"english       
#                 ) 

# vec.stop_words

# vec.fit_transform()

reload(util)

# #stopWords = list(set(stopwords.words('german')))

# vectorizer = TfidfVectorizer(#lowercase=True, #ngram_range=(2, 2),
#                 token_pattern = '(?u)(?:(?!\d)\w)+\\w+', 
#                 analyzer = 'word',  #char_wb
#                 #tokenizer = None, 
#                 stop_words = None)

# X = vectorizer.fit_transform(corpus_train)

# vectorizer.get_feature_names_out().shape


# sim = util.compute_similarity_matrix()
# S = sim.fit_transform(X)

# pr = util.compute_sentence_page_rank()

# pr.fit_transform(S)

from sklearn.pipeline import Pipeline

# Train model:
#
trainer = Pipeline(steps=[
        ('embedding', TfidfVectorizer(token_pattern = '(?u)(?:(?!\d)\w)+\\w+', 
                analyzer = 'word',  #char_wb
                stop_words = None)),
        ('adjacency matrix', util.compute_similarity_matrix()),        
        ('sentence page rank', util.compute_sentence_page_rank())
])
#
scores = trainer.fit_transform(corpus_train)

scores_doc = deepcopy(corpus)
scores_doc['page rank'] = list(scores.values())
ranked_sentence = scores_doc.sort_values(by='page rank', ascending=False, na_position='first')

top_n = 3
summarize_text = ranked_sentence.head(top_n)

print("Top ranked_sentences:\n")
for z,i in enumerate(summarize_text['text'].values): print(z, i)
###############################################################################


#python -m spacy download de

import spacy
nlp = spacy.load('de_core_news_sm')

mywords = "Das ist schon sehr schön mit den Expertinnen und Experten"

for t in nlp.tokenizer(mywords):
    print("Tokenized: %s | Lemma: %s" %(t, t.lemma_))

def transform_texts(texts):
    # Load the annotation models
    #nlp = spacy.load('en')  #English()
    nlp = spacy.load('de_core_news_sm')
    # Stream texts through the models. We accumulate a buffer and release
    # the GIL around the parser, for efficient multi-threading.
    for doc in nlp.pipe(texts, n_threads=4):
        # Iterate over base NPs, e.g. "all their good ideas"
        for np in doc.noun_chunks:
            # Only keep adjectives and nouns, e.g. "good ideas"
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np) > 1:
                # Merge the tokens, e.g. good_ideas
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
            # Iterate over named entities
            for ent in doc.ents:
                if len(ent) > 1:
                    # Merge them into single tokens
                    ent.merge(ent.root.tag_, ent.text, ent.label_)
        token_strings = []
        
        #for token in tokens:
        #    text = token.text.replace(' ', '_')
        #    tag = token.ent_type_ or token.pos_
        #    token_strings.append('%s|%s' % (text, tag))
        #yield ' '.join(token_strings)
        
        return np 