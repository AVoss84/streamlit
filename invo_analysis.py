
from gzip import BadGzipFile
import spacy
from sklearn.pipeline import Pipeline
from copy import deepcopy
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk, re, string, typing        # for type hints
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

pd.set_option('display.max_colwidth', 30

reload(util)
reload(file)

csv = file.CSVService(path="out_df.csv",
                      root_path=Path.home() / "Documents/Arbeit/Allianz/AZVers", delimiter=",")

df = csv.doRead()
print(df.shape)

df.head(1000)
df.info(verbose=True)

corpus = df['de1_eks_postext']

for z, i in enumerate(corpus.values):
    print(z, i)

corpus


#tokens = word_tokenize(se[0])
#tokens = word_tokenize(se)

reload(util)

cleaner = util.clean_text(language='german', without_stopwords=['nicht', 'keine'])


corpus_cl = cleaner.fit_transform(corpus.head(1000))
corpus_cl

corpus_train = corpus_cl['text'].tolist()
corpus_train

for z,(i,k) in enumerate(zip(corpus_train, corpus.head(1000).tolist())):
   print(z, i,"=======",k)



# python -m spacy download de
# python -m spacy download de_core_news_lg

#nlp = spacy.load('de_core_news_sm')
nlp = spacy.load('de_core_news_lg')

dict(nlp)

nlp.pipe

for doc in nlp.pipe(df[col].astype('unicode').values):
    doc_vec = doc.vector                # averaged word embeddings -> doc embedd.
    vectors.append(doc_vec)


documents = corpus.head(1000).tolist()
documents

#mails=['Hallo. Ich spielte am frühen Morgen und ging dann zu einem Freund. Auf Wiedersehen', 'Guten Tag Ich mochte Bälle und will etwas kaufen. Tschüss']

#from spacy.lang.de.examples import sentences 

sent = iter(documents)

document = next(sent)
document


stop_words = set(stopwords.words('german'))

all_cleaned = []
#for z, document in enumerate(documents):
for document in tqdm(documents, total=len(documents)):
    #print(z)
    doc = nlp(document)
    #print(doc.text)     # orig.
    cleaned_text = []
    for token in doc:
         
        #my_token = token.text.lower()
        my_token = token.lemma_.lower()

        # remove punctuation from each word and replace Umlaute
        table = str.maketrans('', '', string.punctuation)          # punctuation
        my_token = util.replace_umlaut(my_token.translate(table))      # Umlaute
        
        # remove remaining tokens that are not alphabetic
        if (not my_token.isalpha()) or (my_token in stop_words):
        #if (not my_token.isalpha()):
            continue
                        
        cleaned_text.append(''.join(my_token))
    #print(cleaned_text)    
    all_cleaned.append(' '.join(cleaned_text))

corpus_cl = pd.DataFrame(all_cleaned, columns=['text']) 
 


print(sentence)
print(doc.text)     # orig.
for token in doc:
    print(token.text, token.pos_, token.dep_, token.lemma_)

    

#res = [(token.text, token.pos_, token.dep_, token.lemma_) for sentence in sentences for token in nlp(sentence)]




my_lemma = []

for mail in test:
     doc = nlp(mail)
     result = ' '.join([x.lemma_ for x in doc]) 
     my_lemma.append(result)

my_lemma




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#n_samples = 2000
#n_features = 1000
n_topics = 20
#n_top_words = 20
batch_size = 128

reload(util)

#cleaner = util.clean_text(language='german', without_stopwords=['nicht', 'keine'])


#corpus_cl = cleaner.fit_transform(corpus.head(5000))
#corpus_cl

#corpus_train = corpus_cl['text'].tolist()


print("Extracting tf features for LDA...")
vec = CountVectorizer(lowercase=True, ngram_range=(5, 10),
                token_pattern = '(?u)(?:(?!\d)\w)+\\w+',
                analyzer = 'char_wb',  #char_wb
                tokenizer = None,
                stop_words = None #"english
                )

vec.stop_words

bag_of_words_vec = vec.fit_transform(corpus_cl['text'])

feature_names = vec.get_feature_names_out()
feature_names

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0
)


LDA = lda.fit(bag_of_words_vec)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 20
display_topics(LDA, feature_names, no_top_words)

#-------------------------------------------------------------------------------------------------------


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=None)

tfidf = tfidf_vectorizer.fit_transform(corpus_cl['text'])

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

from sklearn.decomposition import NMF, LatentDirichletAllocation

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha_W=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

display_topics(nmf, tfidf_feature_names, no_top_words)

