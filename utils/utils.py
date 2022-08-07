import nltk, re, string, typing        # for type hints
from nltk.corpus import stopwords
#from nltk.cluster.util import cosine_distance
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import networkx as nx
#from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel #, rbf_kernel 
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import pairwise_distances
from copy import deepcopy
from tqdm.auto import tqdm
import pandas as pd
from services import file

#nltk.download('punkt')
#nltk.download('stopwords')

class clean_text(BaseEstimator, TransformerMixin):

    def __init__(self, verbose : bool = True, language : str = 'german'):
        self.verbose = verbose
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
    
    def add_stopwords(self, new_stopwords : set):
        return self.stop_words.union(new_stopwords)

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        return self    
    
    def transform(self, X : pd.DataFrame)-> pd.DataFrame:    
        corpus = deepcopy(X)
        cleaned_text = []
        # Preprocess:
        for se in tqdm(corpus.values.tolist(), total=corpus.shape[0]):

            tokens = word_tokenize(se[0])
            
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
