
#from gzip import BadGzipFile
from itertools import count, groupby
from tokenize import group
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

from src.utils import utils as util
from src.services import file
from pathlib import Path

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.max_rows', 500)

reload(util)
reload(file)

# js = file.JSONservice(child_path='data')
# german_stopwords = js.doRead(filename='stopwords.json')


file_name = "Claim descr.csv"

csv = file.CSVService(path=file_name,
                      root_path=Path.home() / "Documents/Arbeit/Allianz/AZVers", delimiter=",")

df = csv.doRead()
print(df.shape)

df.head(1000)
df.info(verbose=True)

col_sel = ['id_sch','invoice_item_id', 'dl_gewerk','firma', 'yylobbez', 'erartbez', 'hsp_eigen', 'hsp_prodbez', 'sartbez', 'sursbez', 'schilderung', 'de1_eks_postext']
col_sel = ['dl_gewerk','de1_eks_postext']
col_sel = ['de1_eks_postext']

corpus = df[col_sel].head(5*10**4)
#corpus = df['de1_eks_postext']
corpus.head(100)
corpus.shape

X = corpus['de1_eks_postext']

# Descriptive stat.:
#corpus.groupby(['id_sch']).agg({'invoice_item_id': 'count'}).rename(columns={'invoice_item_id': '#invoices'})

#corpus.groupby(['dl_gewerk']).agg({'de1_eks_postext': 'count'}).rename(columns={'de1_eks_postext': 'totals'}).sort_values(by='totals', ascending=False).reset_index()

#corpus.groupby(['dl_gewerk','sursbez']).agg({'de1_eks_postext': 'count'}).rename(columns={'de1_eks_postext': 'totals'}).sort_values(by='totals', ascending=False)


#----------------------------
# Preprocessing:
#----------------------------------------------------------------------------

reload(util)

cleaner = util.clean_text(language='german', 
                          without_stopwords=['nicht', 'keine'],
                          with_stopwords=['zzgl'])

corpus_cl = cleaner.fit_transform(X)

corpus_cl.head(20)

#corpus_train = corpus_cl.tolist()
#corpus_train

for z,(i,k) in enumerate(zip(corpus_cl.head(1000).tolist(), corpus.head(1000).tolist())):
   print(z, i,"=======",k)


corpus_0 = corpus.str.lower()

# from typing import TypeVar

# PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

# def remove_whitespace(text : str)-> str:
#     return  " ".join(text.split())

corpus_1 = corpus_0.apply(cleaner.remove_whitespace)

from nltk import word_tokenize

corpus_2 = corpus_1.apply(lambda X: word_tokenize(X))
corpus_2.head(20)

corpus_3 = corpus_2.apply(cleaner.remove_stopwords)
corpus_3.head(20)

#from nltk.tokenize import RegexpTokenizer

# def remove_punct(text):
    
#     tokenizer = RegexpTokenizer(r"\w+")
#     lst=tokenizer.tokenize(' '.join(text))
#     # remove punctuation from each word and replace Umlaute
#     # table = str.maketrans('', '', string.punctuation)          # punctuation
#     # lst = [w.translate(table) for w in text]     # Umlaute

#     return lst


reload(util)
# text = corpus_3.tolist()[1090]
# print(text) 

# remove_punct(text)

corpus_4 = corpus_3.apply(cleaner.remove_punct)
corpus_4.head(20)

text = corpus_4.tolist()[1090]
print(text) 

corpus_5 = corpus_4.apply(cleaner.remove_numbers)
corpus_5.head(20)

corpus_6 = corpus_5.apply(cleaner.remove_digits)
corpus_6.head(20)

corpus_7 = corpus_6.apply(cleaner.remove_non_alphabetic)
corpus_7.head(20)

text = corpus_7.tolist()[1090]
print(text) 

corpus_8 = corpus_7.apply(cleaner.replace_umlaut)   # klappt noch nich!!!!!!!!
corpus_8.head(20)

corpus_9 = corpus_8.apply(cleaner.remove_spec_char_punct)
corpus_9.head(20)

text = corpus_9.tolist()[19]
print(text) 

corpus_10 = corpus_9.apply(cleaner.remove_short_tokens, token_length=4)
#corpus_10 = corpus_9.apply(lambda x: cleaner.remove_short_tokens(x, 3))
corpus_10.head(20)

text = corpus_10.tolist()[1090]
print(text) 

#corpus_11 = corpus_10.apply(cleaner.stem)
#corpus_11.head(20)

corpus_11 = corpus_10.apply(cleaner.untokenize)
corpus_11.head(20)

text = corpus_11.tolist()[1090]
text

nlp = spacy.load('de_core_news_lg')

[nlp(token).lemma_ for token in text]

doc = nlp(text)

[token.lemma_ for token in nlp(text)]

doc = nlp(document)
#print(doc.text)     # orig.
cleaned_text = []
for token in doc:         
    #my_token = token.text.lower()
    my_token = token.lemma_


mails=['Hallo. Ich spielte am frühen Morgen und ging dann zu einem Freund. Auf Wiedersehen', 
'Guten Tag Ich mochte Bälle und will etwas kaufen. Tschüss']

mails_lemma = []

for mail in mails:
     doc = nlp(mail)
     result = ' '.join([x.lemma_ for x in doc]) 
     mails_lemma.append(result)
mails_lemma     
#--------------------------------------------------------------------------------------------


#corpus.groupby(['id_sch']).agg({'invoice_item_id': 'count'}).rename(columns={'invoice_item_id': '#invoices'})

# for z, i in enumerate(corpus.values):
#     print(z, i)

# corpus


#tokens = word_tokenize(se[0])
#tokens = word_tokenize(se)

reload(util)

cleaner = util.clean_text(language='german', 
                          without_stopwords=['nicht', 'keine'],
                          with_stopwords=german_stopwords)

corpus_cl = cleaner.fit_transform(corpus)
corpus_cl

corpus_train = corpus_cl.tolist()
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


# TOPIC
################################################################################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer   # use integer hash instead of actual token in memory

#n_samples = 2000
#n_features = 1000
n_topics = 10
n_top_words = 10
batch_size = 128

reload(util)

cleaner = util.clean_text(language='german', without_stopwords=['nicht', 'keine'])

corpus_cl = cleaner.fit_transform(corpus['de1_eks_postext'])

corpus_cl['y'] = corpus['dl_gewerk']

corpus_cl.head(20)

corpus_cl.shape


# for z,(i,k) in enumerate(zip(corpus_cl.head(1000).tolist(), corpus.head(1000).tolist())):
#    print(z, i,"=======",k)


pipeline = Pipeline([
   #('cleaner', utils.clean_text(verbose=False)),
   ('vectorizer', CountVectorizer(#lowercase=True, 
                ngram_range=(1, 1),
                #token_pattern = '(?u)(?:(?!\d)\w)+\\w+',
                analyzer = 'word',  #char_wb, word
                #tokenizer = None,
                min_df = 0.01, 
                stop_words = cleaner.stop_words #"english
                )),  
    
   ('model', BernoulliNB(alpha = 1))
])

#
print("Extracting tf features for LDA...")
vec = CountVectorizer(#lowercase=True, 
                ngram_range=(1, 1),
                #token_pattern = '(?u)(?:(?!\d)\w)+\\w+',
                analyzer = 'word',  #char_wb, word
                #tokenizer = None,
                min_df = 0.01, 
                stop_words = cleaner.stop_words #"english
                )

bag_of_words_vec = vec.fit_transform(corpus_cl)

feature_names = vec.get_feature_names_out()

for i in feature_names:
 print(i)


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

#
display_topics(LDA, feature_names, no_top_words)

#-------------------------------------------------------------------------------------------------------

# # %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd

reload(util)

pipeline = Pipeline([
   #('cleaner', utils.clean_text(verbose=False)),
   ('vectorizer', CountVectorizer(lowercase=True, #ngram_range=(2, 2),
                ngram_range=(1, 1),
                #token_pattern = '(?u)(?:(?!\d)\w)+\\w+',
                analyzer = 'word',  #char_wb, word
                #tokenizer = None,
                min_df = 0.01, 
                stop_words = cleaner.stop_words #"english
                        )),  
    
   ('model', BernoulliNB(alpha = 1))
])


le = LabelEncoder()

corpus_cl['y_enc'] = le.fit_transform(corpus_cl['y'].tolist())
corpus_cl.head(10)

y = corpus_cl['y_enc']
X = corpus_cl['text']

pipeline.fit(X, y)

pipeline.named_steps['vectorizer'].get_stop_words()
vocab = pipeline.named_steps['vectorizer'].get_feature_names()
print(vocab)

vectorizer = pipeline.named_steps['vectorizer']
nb = pipeline.named_steps['model']

dt = vectorizer.transform(corpus)
doc_term_mat_train = dt.toarray()

joint_abs_freq_train = pd.DataFrame(nb.feature_count_, index=[str(i) for i in nb.classes_], columns=vocab)
joint_abs_freq_train

yhat = pipeline.predict(corpus)

print(doc_term_mat_train)

joint_abs_freq = nb.feature_count_
joint_abs_freq

log_cond_distr = pd.DataFrame(nb.feature_log_prob_, index=[str(i) for i in nb.classes_], columns=vocab)
log_cond_distr


reload(util)

pipe = CountVectorizer(lowercase=True, #ngram_range=(2, 2),
                ngram_range=(1, 1),
                #token_pattern = '(?u)(?:(?!\d)\w)+\\w+',
                analyzer = 'word',  #char_wb, word
                #tokenizer = None,
                min_df = 0.01, 
                stop_words = cleaner.stop_words #"english
                        )

pipe = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100, stop_words=cleaner.stop_words )

nlp_feat = util.make_nb_feat(vectorizer=pipe)
#nlp_feat = util.make_nb_feat(n_features=1000)

pipe = nlp_feat.vectorizer
pipe.fit_transform(X).shape

features_class = nlp_feat.fit_transform(X, y)
features_class.shape

print(len(nlp_feat.vocab_))
print(y.nunique())

features_class
#--------------------------------------------------------------------------------------

# Train model:
#
pipe = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100, stop_words=cleaner.stop_words )

emb = pipe.fit_transform(X)
emb.shape

# trainer = Pipeline(steps=[
#     #('embedding', TfidfVectorizer(token_pattern='(?u)(?:(?!\d)\w)+\\w+',
#     #                              analyzer='word',  # char_wb
#     #                              stop_words=None)),
#     ('embedding', util.make_nb_feat(vectorizer=pipe)),
#     ('adjacency matrix', util.compute_similarity_matrix()),
#     ('sentence page rank', util.compute_sentence_page_rank())
# ])
# #
# scores = trainer.fit_transform(X,y)

# scores_doc = deepcopy(corpus_cl)
# scores_doc.shape
# scores_doc['page rank'] = list(scores.values())

# ranked_sentence = scores_doc.sort_values(by='page rank', ascending=False, na_position='first')

# top_n = 3
# summarize_text = ranked_sentence.head(top_n)

# print("Top ranked_sentences:\n")
# for z, i in enumerate(summarize_text['text'].values):
#     print(z, i, "\n")


from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(emb.toarray())

distances, indices = nbrs.kneighbors(emb.toarray())
distances.shape

adjacency = nbrs.kneighbors_graph(emb.toarray()).toarray()

import networkx as nx
sentence_similarity_graph = nx.from_numpy_array(adjacency)

scores = nx.pagerank(G = sentence_similarity_graph)    # page rank score


# N_UNIVERSES = 5
# N_PLANES = 10
# N_DIMS = emb.shape[1]

# np.random.seed(0)
# planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
#             for _ in range(N_UNIVERSES)]

# reload(util)

# emb.shape
# v = emb.toarray()[0,:]
# v
# planes = planes_l[0]
# planes.shape

# util.hash_value_of_vector(v, planes)

# planes_l

# document_vecs = emb.toarray()

# doc_vecs = document_vecs.tolist()
# len(doc_vecs)


# # Creating the hashtables
# hash_tables = []
# id_tables = []
# for universe_id in range(N_UNIVERSES):  # there are 25 hashes
#     print('working on hash universe #:', universe_id)
#     planes = planes_l[universe_id]
#     hash_table, id_table = util.make_hash_table(doc_vecs, planes)
#     hash_tables.append(hash_table)
#     id_tables.append(id_table)

# doc_id = 0
# vec_to_search = np.array(doc_vecs[doc_id]).reshape(-1, 1)
# vec_to_search.shape

# # Sample
# nearest_neighbor_ids = util.approximate_knn(doc_id, vec_to_search, planes_l, k=3, num_universes_to_use=5, hash_tables=hash_tables, id_tables=id_tables)

#---------------------------------------------------------------------------------------


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 100

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=None)

tfidf = tfidf_vectorizer.fit_transform(corpus_cl['text'])

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

from sklearn.decomposition import NMF, LatentDirichletAllocation

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha_W=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

display_topics(nmf, tfidf_feature_names, no_top_words)


# %%
