import nltk, re, string        # for type hints
from string import punctuation
from nltk.corpus import stopwords
#from nltk.cluster.util import cosine_distance
from nltk.stem.snowball import SnowballStemmer
#from pydantic import BaseModel
from nltk.tokenize import word_tokenize, RegexpTokenizer
import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel #, rbf_kernel 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer   # use integer hash instead of actual token in memory
#from sklearn.metrics import pairwise_distances
from typing import List
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
from src.services import file
import spacy

#nltk.download('punkt')
#nltk.download('stopwords')

class clean_text(BaseEstimator, TransformerMixin):

    def __init__(self, verbose : bool = True, language : str = 'german', **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs
        self.stop_words = set(stopwords.words(language))
        if self.verbose: print(f'Using {len(self.stop_words)} stop words.')
        self.german_stopwords = file.JSONservice(verbose=False).doRead(filename='stopwords.json')
        if self.verbose: print(f'Adding custom German stop words...')
        self.stop_words = self._add_stopwords(self.german_stopwords) 

        if 'without_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._remove_stopwords(self.kwargs.get('without_stopwords', ''))
                
        if 'with_stopwords' in list(self.kwargs.keys()):
            self.stop_words = self._add_stopwords(self.kwargs.get('with_stopwords', '')) 

        self.stemmer = SnowballStemmer(language)
        self.nlp = spacy.load('de_core_news_lg')
        self.umlaut = file.YAMLservice(child_path = "/config").doRead(filename = "preproc_txt.yaml")

    def _add_stopwords(self, new_stopwords : list)-> set:
        """
        Change stopword list. Include into stopword list

        Args:
            new_stopwords (list): _description_

        Returns:
            set: _description_
        """
        old = self.stop_words.copy()
        self.stop_words = self.stop_words.union(set(new_stopwords))
        if self.verbose: print(f"Added {len(self.stop_words)-len(old)} stopword(s).")
        return self.stop_words

    def _remove_stopwords(self, without_stopwords : list)-> set:
        """
        Change stopword list. Exclude from stopwords

        Args:
            without_stopwords (list): _description_

        Returns:
            set: _description_
        """
        old = self.stop_words.copy()
        self.stop_words = self.stop_words.difference(set(without_stopwords))
        if self.verbose: print(f"Removed {len(old)-len(self.stop_words)} stopword(s).")
        return self.stop_words

    def untokenize(self, text: List[str])-> str:
        """Revert tokenization: list of strings -> string"""
        return " ".join([w for w in text])

    def count_stopwords(self):
        print(f'{len(self.stop_words)} used.')
 
    def remove_whitespace(self, text : str)-> str:
        return  " ".join(text.split())

    def remove_punctuation(self, text: str)-> str:    
       return [re.sub(f"[{re.escape(punctuation)}]", "", token) for token in text]

    def remove_numbers(self, text: str)-> str:    
       return [re.sub(r"\b[0-9]+\b\s*", "", token) for token in text]

    def remove_stopwords(self, text : str)-> str:
        return [token for token in text if token not in self.stop_words]

    def remove_digits(self, text: str)-> str: 
        """Remove digits instead of any number, e.g. keep dates"""
        return [token for token in text if not token.isdigit()]

    def remove_non_alphabetic(self, text: str)-> str: 
        """Remove non-alphabetic characters"""
        return [token for token in text if token.isalpha()]
    
    def remove_spec_char_punct(self, text: str)-> str: 
        """Remove all special characters and punctuation"""
        return [re.sub(r"[^A-Za-z0-9\s]+", "", token) for token in text]

    def remove_short_tokens(self, text: str, token_length : int = 2)-> str: 
        """Remove short tokens"""
        return [token for token in text if len(token) > token_length]

    def remove_punct(self, text: str)-> str:
        tokenizer = RegexpTokenizer(r"\w+")
        lst = tokenizer.tokenize(' '.join(text))
        # table = str.maketrans('', '', string.punctuation)          # punctuation
        # lst = [w.translate(table) for w in text]     # Umlaute
        return lst

    def replace_umlaut(self, text : str) -> str:
        """Replace special German umlauts (vowel mutations) from text"""
        vowel_char_map = {ord(k): v for k,v in self.umlaut['replace']['german']['umlaute'].items()}  # use unicode value of Umlaut
        return [token.translate(vowel_char_map) for token in text]

    def stem(self, text : str)-> str:
        return [self.stemmer.stem(w)  for w in text]
    
    def lemmatize(self, text : str)-> str:
        text = self.untokenize(text)
        return [token.lemma_ for token in self.nlp(text)]

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        return self    
    
    def transform(self, X : pd.Series, **param)-> pd.Series:    
        corpus = deepcopy(X)
        corpus = corpus.str.lower()
        corpus = corpus.apply(self.remove_whitespace)
        corpus = corpus.apply(lambda x: word_tokenize(x))
        corpus = corpus.apply(self.remove_stopwords)
        corpus = corpus.apply(self.remove_punct)
        corpus = corpus.apply(self.remove_numbers)
        corpus = corpus.apply(self.remove_digits)
        corpus = corpus.apply(self.remove_non_alphabetic)
        corpus = corpus.apply(self.replace_umlaut)   
        corpus = corpus.apply(self.remove_spec_char_punct)
        corpus = corpus.apply(self.remove_short_tokens, token_length=3)
        corpus = corpus.apply(self.stem)
        #corpus = corpus.apply(self.lemmatize)   # makes preprocessing very slow though
        corpus = corpus.apply(self.untokenize)
        if self.verbose: print("Finished preprocessing.")
        return corpus #.to_frame(name="text") 


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
    Calculate similarity matrix / adjacency matrix
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



class make_nb_feat(BaseEstimator, TransformerMixin):
  """
  Create 'Naive Bayes'- like document embeddings
  """
  def __init__(self, verbose : bool = True, vectorizer : TransformerMixin = None, **vect_param):
        
        self.verbose = verbose  
        self.vectorizer = vectorizer
        if verbose : print('-- Creating Naive Bayes like document embeddings --\n')  
        if self.vectorizer is None:      
            # self.vectorizer = CountVectorizer(analyzer = 'word',  
            #                     stop_words = None, **vect_param)
            # Compute raw counts using hashing vectorizer 
            self.vectorizer = HashingVectorizer(analyzer = 'word',    # Small numbers of n_features can cause hash collisions 
                                                alternate_sign=False, **vect_param)
        self.pipeline = Pipeline([
                #('cleaner', utils.clean_text(verbose=False)),   # we assume X is already preprocessed at the moment
                ('vectorizer', self.vectorizer),  
                ('model', BernoulliNB(alpha=1))
                ])

    
  def fit(self, X, y):
        
      self.pipeline.fit(X, y)
      self.pipeline.named_steps['vectorizer'].get_stop_words()
      try:
          self.vocab_ = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
      except Exception as ex:
          self.vocab_ = None  
      self.vectorizer = self.pipeline.named_steps['vectorizer']
      self.model = self.pipeline.named_steps['model']
      dt = self.vectorizer.transform(X)   # train set
      self.doc_term_mat_train = dt.toarray()
      self.log_cond_distr_train = pd.DataFrame(self.model.feature_log_prob_, index=[str(i) for i in self.model.classes_], columns=self.vocab_)
      #self.joint_abs_freq_train = pd.DataFrame(self.model.feature_count_, index=[str(i) for i in self.model.classes_], columns=self.vocab_)
      return self

  def transform(self, X)-> pd.DataFrame:

      dt = self.vectorizer.transform(X)
      self.doc_term_mat = dt.toarray()
      features_class = pd.DataFrame()
      for c in self.model.classes_:
            # log class cond. prob
            feat_c = np.sum(self.doc_term_mat * self.log_cond_distr_train.loc[str(c),:].values, axis = 1)   # broadcast
            # Joint abs. freq
            #feat_c = np.sum(self.doc_term_mat * self.joint_abs_freq_train.loc[str(c),:].values, axis = 1)   # broadcast
            features_class['level'+str(c)] = feat_c
      return features_class 


def nearest_neighbor(v, candidates, k=1):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v,row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)
        
    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]
    return k_idx


def hash_value_of_vector(v, planes):
    """Create a hash for a vector; hash_id says which random hash to use.
    Input:
        - v:  vector of tweet. It's dimension is (1, N_DIMS)
        - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
    Output:
        - res: a number which is used as a hash for your vector

    """
    # for the set of planes,
    # calculate the dot product between the vector and the matrix containing the planes
    # remember that planes has shape (300, 10)
    # The dot product will have the shape (1,10)
    dot_product = np.dot(v,planes)
    
    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)
    
    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    h = sign_of_dot_product>=0

    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = np.squeeze(h)

    # initialize the hash value to 0
    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2,i)*h[i]

    # cast hash_value as an integer
    hash_value = int(hash_value)

    return hash_value


def make_hash_table(vecs, planes):
    """
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    """
    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2**num_of_planes

    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i:[] for i in range(num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i:[] for i in range(num_buckets)}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):

        # calculate the hash value for the vector
        h = hash_value_of_vector(v,planes)
        #print(h)
        #print('******')
        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v)

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i)

    return hash_table, id_table



def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=25, hash_tables = [], id_tables = []):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= num_universes_to_use

    # Vectors that will be checked as p0ossible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]
                
                # append the new_id (the index for the document) to the list of ids to consider
                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)
                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)
    print(nearest_neighbor_idx_l)
    print(ids_to_consider_l)
    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids