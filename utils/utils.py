import nltk, re, string, typing        # for type hints
from nltk.corpus import stopwords
#from nltk.cluster.util import cosine_distance
from nltk.stem.snowball import SnowballStemmer
#from pydantic import BaseModel
from nltk.tokenize import word_tokenize
import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel #, rbf_kernel 
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import pairwise_distances

import gensim
from gensim.models import FastText, Phrases, phrases, TfidfModel
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
from gensim.parsing.preprocessing import stem_text, strip_multiple_whitespaces, strip_short, strip_non_alphanum, strip_punctuation, strip_numeric

from copy import deepcopy
from tqdm.auto import tqdm
import pandas as pd
from services import file
import spacy

#nltk.download('punkt')
#nltk.download('stopwords')

class clean_text(BaseEstimator, TransformerMixin):

    def __init__(self, verbose : bool = True, language : str = 'german', **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs

        self.stop_words = set(stopwords.words(language))
        if list(self.kwargs.keys())[0] == 'without_stopwords':
            self.stop_words = self._remove_stopwords(self.kwargs)
                
        if list(self.kwargs.keys())[0]  == 'new_stopwords':
            self.stop_words = self._add_stopwords(self.kwargs) 

        self.stemmer = SnowballStemmer(language)
        self.nlp = spacy.load('de_core_news_lg')
    
    def _add_stopwords(self, new_stopwords : list):
        if self.verbose: print(f"Added {len(new_stopwords)} stop word(s).")
        self.stop_words = self.stop_words.union(set(new_stopwords))
        return self.stop_words

    def _remove_stopwords(self, without_stopwords : list):
        if self.verbose: print(f"Removed {len(without_stopwords)} stop word(s).")
        self.stop_words = self.stop_words.difference(set(without_stopwords))
        return self.stop_words

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        return self    
    
    def transform(self, X : pd.DataFrame)-> pd.DataFrame:    
        corpus = deepcopy(X)
        cleaned_text = []
        # Preprocess:
        for se in tqdm(corpus.values.tolist(), total=corpus.shape[0]):

            #tokens = word_tokenize(se)
            tokens = [token.text for token in self.nlp(se)]

            # convert to lower case
            tokens = [w.lower() for w in tokens]

            # remove punctuation from each word and replace Umlaute
            table = str.maketrans('', '', string.punctuation)          # punctuation
            stripped = [replace_umlaut(w.translate(table)) for w in tokens]     # Umlaute

            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]   
            
            # filter out stop words and apply stemming:
            words = [self.stemmer.stem(w)  for w in words if not w in self.stop_words]
            cleaned_text.append(' '.join(words))
        return pd.DataFrame(cleaned_text, columns=['text']) 


def replace_umlaut(mystring : str) -> str:
    """
    Replace special German umlauts (vowel mutations) from text
    """
    repl_dict = file.YAMLservice(child_path = "config").doRead(filename = "preproc_txt.yaml")
    vowel_char_map = {ord(k): v for k,v in repl_dict['replace']['german']['umlaute'].items()}  # use unicode value of Umlaut
    return mystring.translate(vowel_char_map)


class text_tools:
    def __init__(self, verbose : bool = True):
        self.verbose = verbose
    
    @staticmethod
    def iter_document(corpus : list)-> tuple:
        """_
        Generator that returns a single document in each call. 
        Output: string
        """
        i, k = 0, len(corpus) 
        while i < k:
            yield i, corpus[i]
            i += 1 


    def read_article(self, path: str) -> pd.DataFrame:
        """_
        Args:
            path (str): relative path with filename

        Returns:
            list: _description_
        """
        try:
            with open(path, "r", errors='ignore') as file:
                filedata = file.readlines()       
            if self.verbose : print(f'Successfully imported corpus of size {len(filedata)}')    
            return pd.DataFrame(filedata, columns=['text'])    
        except Exception as ex:
            print(ex)


class compute_similarity_matrix(BaseEstimator, TransformerMixin):
    """
    Calculate similarity matrix
    """ 
    def __init__(self, verbose : bool = True):
        self.verbose = verbose
        if self.verbose : print('-- Compute similarity matrix --')

    def fit(self, X : np.ndarray, y : np.ndarray = None):
        return self    
    
    def transform(self, X : np.ndarray)-> np.ndarray:  
        #return = cosine_similarity(X = sentence_embeddings)    # equivalent to linear dot prod. kernel in case of tf-idf, as already normalized but slower!
        return linear_kernel(X = X)    # kernel = similarity ; vs. distance metric 

class compute_sentence_page_rank(BaseEstimator, TransformerMixin):
    """
    Fit undirected graphical model based on sentence (=document) similarities (Adjacency matrix) 
    and compute page rank via based on Markov chain - eigenvectors of adj matrix
    """
    def __init__(self, verbose : bool = True):
        self.verbose = verbose
        if self.verbose : print('-- Compute sentence page rank --')

    def fit(self, X : np.ndarray, y : np.ndarray = None):
        return self    
    
    def transform(self, X : np.ndarray)-> np.ndarray:    
        # Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(X)
        scores = nx.pagerank(G = sentence_similarity_graph)    # page rank score
        return scores


# def clean_text_OLD(text, for_embedding=False):
#     """
#         - remove any html tags (< /br> often found)
#         - Keep only ASCII + European Chars and whitespace, no digits
#         - remove single letter chars
#         - convert all whitespaces (tabs etc.) to single wspace
#         if not for embedding (but e.g. tdf-idf):
#         - all lowercase
#         - remove stopwords, punctuation and stemm
#     """
#     RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
#     RE_TAGS = re.compile(r"<[^>]+>")
#     RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
#     RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
#     if for_embedding:
#         # Keep punctuation
#         RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
#         RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

#     text = re.sub(RE_TAGS, " ", text)
#     text = re.sub(RE_ASCII, " ", text)
#     text = re.sub(RE_SINGLECHAR, " ", text)
#     text = re.sub(RE_WSPACE, " ", text)

#     word_tokens = word_tokenize(text)
#     words_tokens_lower = [word.lower() for word in word_tokens]

#     if for_embedding:
#         # no stemming, lowering and punctuation / stop words removal
#         words_filtered = word_tokens
#     else:
#         words_filtered = [
#             stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
#         ]

#     text_clean = " ".join(words_filtered)
#     return text_clean


class embeddings(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        
        self.language = kwargs['language']
        self.epochs = kwargs['epochs']
        self.columns = kwargs['columns']
        self.model_type = kwargs['model_type']
        self.size = kwargs['size']
        self.window = kwargs['window']
        self.save_model = kwargs['save_model']
        self.min_count = kwargs['min_count']
        self.n_grams = kwargs['n_grams']
        self.threshold = kwargs['threshold']
        self.lemmatize = kwargs['lemmatize']
        super(embeddings, self).__init__()
        print("Initialize vectorizer.")
        
    def generate_dataframe(self, vectors, column):
        vectors = pd.DataFrame.from_records(vectors, columns=[column + '_{}'.format(i) for i in range(len(vectors[0]))])
        return vectors
            
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_stopwords(self, texts, my_stop_words):
        return [[word for word in simple_preprocess(str(doc)) if word not in my_stop_words] for doc in texts]

    def remove_shorts(self, texts, minsize = 4):
        return [[strip_short(word, minsize = minsize) for word in doc if len(strip_short(word, minsize = minsize)) > 0] for doc in texts]

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def make_trigrams(self, texts):
        return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out        

    def create_dictionary(self, docs):
        """Create dictionary and corpus (Term Document Frequency (document to bag-of-words))"""
        
        # Create Dictionary
        self.dictionary = corpora.Dictionary(docs)      # id2word
        # Term Document Frequency (document to bag-of-words)
        self.corpus = [self.dictionary.doc2bow(text) for text in docs]
        self.corpus_with_word_counts = [[(self.dictionary[id], count) for id, count in line] for line in self.corpus]
        return self

    def _make_document_vectors(self, normalize_weights = False):
        
        """Calculate weighted average of word vectors per document 
           using the normalized TF-IDF weights.
        """
        vec_map = list(); avg_vectors = np.empty((len(self.corpus[-len(self.docs_new):]),self.model.vector_size))
        for i, doc in enumerate(self.tfidf[self.corpus[-len(self.docs_new):]]):
            weight = []; vec=np.empty((self.model.vector_size,0))    # zero column!!
            for id, freq in doc:
                    weight.append(np.round(freq,4))
                    v = np.array(self.model.wv.get_vector(self.dictionary[id]))  
                    vec = np.column_stack((vec, v))
            weight_avg = np.zeros((1,self.model.vector_size))
            if len(weight) > 0:
                if normalize_weights:
                    wei = np.array(weight/sum(weight)).reshape(len(weight),1)  
                else:
                    wei = np.array(weight).reshape(len(weight),1) 
                #sp_vec = csr_matrix(vec)
                #sp_wei = csr_matrix(wei)
                weight_avg = np.matmul(vec, wei)
                #weight_avg = sp_vec.dot(sp_wei)
            vec_map.append((i, self.training_input[i], weight_avg)) 
            avg_vectors[i,:] = weight_avg.reshape(1,self.model.vector_size)
            #avg_vectors = hstack([avg_vectors, weight_avg]).toarray()
        return vec_map, avg_vectors

    
    def text_preprocess(self, documents, n_grams = 2, threshold = 100, min_count = 5, lemmatize = False, **phraser):
    
        """
        see https://radimrehurek.com/gensim/models/phrases.html for details
        min_count – Ignore all words and bigrams with total collected count lower than this value.
        threshold – Represent a score threshold for forming the phrases (higher means fewer phrases). 
                    A phrase of words a followed by b is accepted if the score of the phrase is greater 
                    than threshold. 
                    Heavily depends on concrete scoring-function.
        """    
        # Tokenize documents/sentences in corpus:
        data_words = list(self.sent_to_words(documents))
        self.stop_words = set(stopwords.words(self.language))
        
        bigram = Phrases(data_words, min_count = min_count, threshold = threshold, 
                         common_terms = self.stop_words, **phraser) # higher threshold fewer phrases.
        trigram = Phrases(bigram[data_words], threshold = threshold, common_terms = self.stop_words, **phraser)  
        self.bigram_mod = phrases.Phraser(bigram)
        self.trigram_mod = phrases.Phraser(trigram)
            
        # Remove Stop Words and short words
        data_words = self.remove_stopwords(data_words, self.stop_words);print("Remove stopwords.")
        data_words = self.remove_shorts(data_words);print("Remove short tokens.")
        
        if n_grams == 1:
            print("Building 1-grams.")
            data_words = data_words
        elif n_grams == 2:
            print("Building 2-grams.")
            data_words = self.make_bigrams(data_words)
        else:    
            print("Building 3-grams.")
            data_words = self.make_trigrams(data_words)
        
        if lemmatize:
            print("Do lemmatization keeping only noun, adj, vb, adv.")
            #self.nlp = spacy.load('en', disable=['parser', 'ner'])
            self.nlp = spacy.load('de_core_news_lg')
            data_words = self.lemmatization(self.data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        try:
            self.data_words.extend(data_words)
        except AttributeError:
            self.data_words = data_words
        
        return data_words
    
    
    def build_vecm(self, texts=None, 
                   model_type = ['word2vec', 'doc2vec', 'fasttext'],
                   save_model = dict(save = False, model_name  = "my_text2vec"),  
                   **vec_para):
        
        assert len(model_type)>1, 'Choose only one model_type!'
        self.model_type = model_type
        #self.train_tf_idf = train_tf_idf
        
        if texts is None:
           texts = self.data_words
        inputs = self.create_dictionary(docs = texts)
        corpus = inputs.corpus
        
        self.save_model = save_model
        self.fname = get_tmpfile(glob.UC_CODE_DIR+"predictor_fraud/resources/models/"+self.save_model['model_name'])
        
        if self.model_type == 'doc2vec':
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
            self.model = Doc2Vec(**vec_para)
            print("Doc2vec initialized!")
            self.model.build_vocab(documents)
            self.used_vocabulary = self.model.wv.vocab
            print("Vocabulary has been built. Size:", len(self.used_vocabulary))
            self.training_input = documents
            
        elif self.model_type == 'fasttext':
            self.model = FastText(**vec_para)
            print("FastText initialized!")    
            self.model.build_vocab(sentences = texts)
            self.used_vocabulary = self.model.wv.vocab
            print("Vocabulary has been built. Size:", len(self.used_vocabulary))
            self.training_input = texts   
            
        else :    
            self.model = Word2Vec(**vec_para)
            print("Word2Vec initialized!")     
            self.model.build_vocab(sentences = texts)
            self.used_vocabulary = self.model.wv.vocab
            print("Vocabulary has been built. Size:", len(self.used_vocabulary))
            self.training_input = texts 
    
    
    def fit(self, X=None, y=None, **vec_para):
        
        X = X.values.tolist()
        
        """Train language model"""
        self.text_preprocess(X, n_grams = self.n_grams, threshold = self.threshold, min_count = 30, 
                             lemmatize = self.lemmatize)
        self.build_vecm(model_type = self.model_type, size=self.size, window=self.window, workers=cpu_count()-1, 
                        save_model = self.save_model, min_count = self.min_count)
        self.model.epochs = self.epochs
        
        if X is not None:
            self.training_input = X
                
        if self.model_type is not 'doc2vec':                
            print("Train Tf-Idf model to obtain term weights...")
            self.tfidf = TfidfModel(self.corpus, smartirs='ntc', normalize = True)        
            print("Finished.")   
                
        t = time() ; print("Start training vector model...")
        self.model.train(self.training_input, total_examples = self.model.corpus_count, 
                         epochs = self.epochs, **vec_para)    
        print('Time to train the model: {} mins'.format(round((time() - t)/60, 2)))
        self.model.init_sims(replace=True)
        
        if self.save_model['save']: 
            self.model.save(self.fname) ; print(self.model_type,"model saved to:", self.fname)
        return self
    
   
    def transform(self, X=None, **kwargs):
        
        X = X.values.tolist()
        infer_vectors = None
                
        if (X is not None) & isinstance(X, list):
            self.docs_new = self.text_preprocess(X, n_grams = self.n_grams, threshold = self.threshold, 
                                                 min_count = 30, 
                                                 lemmatize = self.lemmatize)
        else:
            self.docs_new = self.training_input    
            
            
        if self.model_type == 'doc2vec':
            self.model = Doc2Vec.load(self.fname)  # you can continue training with the loaded model!
            
            # Update the model with new data
            #self.model.build_vocab(docs_new, update=True)
            #self.model.train(docs_new, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            
            # Infer embedding vector for new data:
            print("Infer word embeddings for input corpus.")
            infer_vectors = pd.Series(self.docs_new).apply(self.model.infer_vector)
            vectors = np.array(infer_vectors.tolist())
            
        elif self.model_type == 'fasttext':
            self.model = FastText.load(self.fname)
            print("Language model loaded from:", self.fname)
            self.model.build_vocab(self.docs_new, update=True)
            self.model.train(self.docs_new, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.create_dictionary(self.data_words)
            _, vectors = self._make_document_vectors()

        else :
            self.model = Word2Vec.load(self.fname)  
            print("Language model loaded from:", self.fname)
            self.model.build_vocab(self.docs_new, update=True)
            self.model.train(self.docs_new, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.create_dictionary(self.data_words)
            _, vectors = self._make_document_vectors()
        
        vectors = self.generate_dataframe(vectors, self.columns[0])
        self.word_embedding_names = vectors.columns
        return vectors