import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pymystem3 import Mystem
import plotly.express as px
import plotly.graph_objects as go
from time import time
from IPython.display import clear_output
from sys import getsizeof

import warnings
warnings.filterwarnings('ignore')

UKR_CHANNELS = [
    'Труха⚡️Украина', 'Лачен пишет', 'Украинская правда. Главное',
    'Вы хотите как на Украине?', 'Борис Філатов', 'RAGNAROCK PRIVET',
    'УНИАН - новости Украины | война с Россией | новини України | війна з Росією',
    'Украина 24/7 Новости | Война | Новини', 'Быть Или',
    'Украина Сейчас: новости, война, Россия'
]

UKR_LETTERS = ['ї', 'є', 'ґ', 'і']

CHEAT_WORDS = [
    '03', '04', '05', '1378', '2022', '3801', '3806', '4149', '4276',
    '4279', '9521', '9842', 'akimapachev', 'amp', 'anna', 'com',
    'daily', 'diza', 'donbass', 'epoddubny', 'https', 'index', 'me',
    'news', 'opersvodki', 'pravda', 'rus', 'rvvoenkor', 'sashakots',
    'ua', 'wargonzo', 'www', 'www pravda', 'мид', 'труха', 'труха украина',
    'украина сейчас', 'pravda com', 'daily news', 'com ua', 'https www',
    'me rvvoenkor', 'rus news', 'ua rus', 'wargonzo наш'
]

def time_decorator(function):
    """
    Just a decorator for printing the timings.
    """
    from time import time
    def inner(*args, **kwargs):
        start = time()
        result = function(*args, **kwargs)
        elapsed_time = round(time() - start, 2)
        output = f'{function.__name__} took {elapsed_time} seconds.'
        print(output)
        return result
    return inner

class Preprocessor:
    
    def __init__(self, data=None):
        """
        A class for the preprocessing purposes. Main methods icnludes:
        reading, cleaning, lemmatizing and vectorizing the data.
        """
        self.data = data
        self.lemmas = None
        self.X_train = None
        self.X_test = None
        self.ukr_train = None
        self.ukr_test = None
        self.channel_train = None
        self.channel_test = None
        self.percent_ukr = 0
        self.percent_rus = 1
        self.lemmatized = False
        self.vectorized = False
        self.cheat_words = CHEAT_WORDS
    
    @time_decorator
    def read_data(self, filename='random_msgs.csv', sep='¶∆',
                  header=None):
        """
        Reads the csv file into 4 columns:
        channel
        date of publication
        message
        ukrainian - 1 if ukrainian channel, 0 - otherwise.
        """
        if self.data is None:
            self.data = pd.read_csv(filename, sep=sep, header=header)
            self.data.columns = ['channel', 'date', 'msg']
            self.data['ukrainian'] = self.data['channel'].\
            apply(lambda x: 1 if x in UKR_CHANNELS else 0)
            self.data['ukrainian'] = self.data['ukrainian'].astype('int8')
            self.data = self.data[self.data['channel'] != 'вечеряємо']
            self.percent_ukr = self.data['ukrainian'].mean()
            self.percent_rus = 1 - self.percent_ukr
    
    def get_data(self):
        """
        Method to get the df.
        """
        return self.data
    
    def get_percents_ukr_rus(self):
        """
        Method to get the percentage of ukrainian and russian messages among
        the dataset.
        """
        return self.percent_ukr, self.percent_rus
    
    @time_decorator
    def preprocess(self, remove_ukr_msgs=True, cut_less_than=18):
        """
        This method:
        removes short messages (with less than 18 characters);
        removes messages with ukrainian letters.
        """
        if remove_ukr_msgs:
            for letter in UKR_LETTERS:
                self.data = self.data[self.data['msg'].str.lower().\
                                        str.contains(letter) == False]
        self.data = self.data[self.data['msg'].str.len() > cut_less_than]
        self.data = self.data.reset_index(drop=True)
        self.percent_ukr = self.data['ukrainian'].mean()
        self.percent_rus = 1 - self.percent_ukr
    
    @time_decorator
    def lemmatize(self, *sentences):
        """
        This method has 2 usages:
        internal; i.e. to lemmatize all messages in the dataset. Runs about 2.5
        minutes.
        outside; to lemmatize a given sequence of sentences.
        """
        mystem = Mystem()
        if not sentences:
            if not self.lemmatized:
                def preprocess_text(text):
                    tokens = mystem.lemmatize(text.lower())
                    text = " ".join(tokens)
                    return text

                self.data['msg'] = self.data['msg'].apply(preprocess_text)
                self.lemmas = self.data['msg'].copy()
                self.lemmatized = True
        else:
            result = []
            for sentence in sentences:
                tokens = mystem.lemmatize(sentence.lower())
                result.append(' '.join(tokens))
            return result
    
    def get_lemmas(self):
        """
        Method to get lemmatized messages.
        """
        return self.lemmas
    
    def train_test_split(self, random_state=1, train_size=.8):
        """
        This method clones scikit-learn train_test_split.
        """
        self.X_train, self.X_test, self.ukr_train, self.ukr_test,\
        self.channel_train, self.channel_test = \
        train_test_split(
            self.data['msg'], self.data['ukrainian'], self.data['channel'],
            random_state=random_state, train_size=train_size
        )
    
    def get_train_test_split(self):
        """
        Returns the train and test part.
        """
        return self.X_train, self.X_test, self.ukr_train, self.ukr_test,\
        self.channel_train, self.channel_test
    
    @time_decorator
    def vectorize(self, ngram_range=(1,1), sublinear_tf=True, binary=False):
        """
        This method creates a pipeline of CountVectorizer() and TfidfTransformer().
        If CountVectorizer is needed - use count_transform method.
        If TfidfVectorizer is needed - just call a tfidf_transform method.
        """
        try:
            if not self.vectorized:
                self.tfidf = Pipeline([
                    ('vect', CountVectorizer(binary=binary, ngram_range=ngram_range)),
                    ('tfidf', TfidfTransformer(sublinear_tf=sublinear_tf))
                ]).fit(self.X_train)
                self.vect = self.tfidf['vect']
                self.vectorized = True
        except TypeError:
            print("You didn't initialize data or train_test_split.")
        
    
    def get_vectorizer(self, tfidf=True):
        """
        Returns the actual vectorizer.
        """
        return self.vectorizer
    
    @time_decorator
    def tfidf_transform(self):
        """
        Applies TfidfTransform to data.
        """
        try:
            self.vectorizer = self.tfidf
            X_train = self.X_train = self.vectorizer.transform(self.X_train).T
            X_test =  self.X_test = self.vectorizer.transform(self.X_test).T
            return X_train, X_test
        except AttributeError:
            print("You didn't initialize read_data, train_test_split or vectorize.")
    
    @time_decorator
    def count_transform(self):
        """
        Applies CountTransform to data.
        """
        try:
            self.vectorizer = self.vect
            X_train = self.X_train = self.vectorizer.transform(self.X_train).asfptype().T
            X_test = self.X_test = self.vectorizer.transform(self.X_test).asfptype().T
            return X_train, X_test
        except AttributeError:
            print("You didn't initialize read_data, train_test_split or vectorize.")
    
    @time_decorator
    def remove_cheat_words(self, method='manual', freq_pivot=.5,
                           cheat_words=CHEAT_WORDS):
        """
        Removes cheat_words, like channel tags, social media links or
        authors names.
        """
        if method == 'manual':
            delete_mask = np.zeros(self.X_train.shape[0], dtype=bool)
            delete_mask[np.isin(np.array(
                    self.vectorizer.get_feature_names_out()), cheat_words)
            ] = True
            self.X_train = self.X_train.T[:, ~delete_mask].T
            self.X_test = self.X_test.T[:, ~delete_mask].T
            self.delete_mask = delete_mask
            self.cheat_words = np.array(
                self.vectorizer.get_feature_names_out()
            ).T[delete_mask]
        else:
            delete_mask = np.zeros(self.X_train.shape[0], dtype=bool)
            for channel in self.channel_trainchannel_train.unique():
                arr = self.X_train.T[self.channel_train == channel]
                delete_mask |= np.array((np.sum(arr > 0, axis=0) / arr.shape[0]) > .5)[0]

            self.X_train = self.X_train.T[:, ~delete_mask].T
            self.X_test = self.X_test.T[:, ~delete_mask].T
            self.delete_mask = delete_mask
            self.cheat_words = np.array(
                self.vectorizer.get_feature_names_out()
            ).T[delete_mask]

    def get_cheat_words(self):
        """
        Returns the deleted cheat_words.
        """
        return self.cheat_words
    
    def get_delete_mask(self):
        """
        Returns the mask of cheat_words, which can be applied onto vectorizer matrix.
        """
        return self.delete_mask

class Predictor:
    
    def __init__(self, SVD=[None, None, None]):
        """
        Predictor class, which contains 3 predicting methods.
        """
        self.Terms, self.S, self.Documents = SVD
        if not self.S:
            self.calculated_svd = False
        else:
            self.calculated_svd = True
    
    def get_SVD(self):
        """
        Returns SVD if it is calculated onde.
        """
        if self.calculated_svd:
            return self.Terms, self.S, self.Documents
        return 'You need to calculate SVD first'
    
    @time_decorator
    def train_LSA(self, X_train, ukr_train, k=150):
        """
        Calculates the SVD and then finds the centre of ukrainian and russian
        clouds.
        """
        if not self.calculated_svd:
            self.Terms, self.S, self.Documents = svds(X_train, k=k)
            self.ukr_centre = np.array([np.mean(self.Documents.T[ukr_train == 1], axis=0)])
            self.rus_centre = np.array([np.mean(self.Documents.T[ukr_train == 0], axis=0)])
            self.calculated_svd = True
    
    @time_decorator
    def predict_LSA(self, X_pred):
        """
        Projects X_pred onto orthonormal basis Terms and then scales in each axis by S.
        """
        Documents_pred = np.diag(1 / self.S) @ self.Terms.T @ X_pred
        dist_to_ukr = cdist(self.ukr_centre, Documents_pred.T, metric='euclidean')[0]
        dist_to_rus = cdist(self.rus_centre, Documents_pred.T, metric='euclidean')[0]
        ukr_pred = self.ukr_pred = np.array([dist_to_ukr < dist_to_rus]).reshape((-1, 1))
        return ukr_pred
    
    def evaluate(self, ukr_test):
        """
        Method to evaluate the prediction. Returns the ratio between correct guesses and
        total.
        """
        ukr_test = np.array(ukr_test).astype(bool).reshape((-1, 1))
        self.accuracy = round(100 * np.sum(self.ukr_pred == ukr_test) / len(ukr_test), 2)
        return self.accuracy
    
    def train_NBC(self, X_train, ukr_train, percent_ukr=0):
        """
        Naive Bayes Classifier. Need to use only CountVectorizer(binary=True)
        for calculating the relative frequency.
        """
        self.terms_prob_ukr = np.mean(X_train.T[ukr_train == 1], axis=0)
        self.terms_prob_rus = np.mean(X_train.T[ukr_train == 0], axis=0)
        self.percent_ukr = percent_ukr
        self.percent_rus = 1 - percent_ukr
    
    def predict_NBC(self, X_pred):
        """
        A method, which predicts, using trained Naive Bayes Classifier.
        """
        self.ukr_prob = self.percent_ukr * X_pred.T * self.terms_prob_ukr.T
        self.rus_prob = self.percent_rus * X_pred.T * self.terms_prob_rus.T
        ukr_pred = self.ukr_pred = self.ukr_prob > self.rus_prob

    def train_LR(self, X_train, ukr_train):
        """
        Trains Logistic Regressin.
        """
        self.logistic_regression = LogisticRegression(random_state=1).fit(X_train.T, ukr_train)
    
    def predict_LR(self, X_pred):
        """
        Predicts, using pre-trained logistic model.
        """
        ukr_pred = self.ukr_pred = np.array([self.logistic_regression.predict(X_pred.T)]).reshape((-1, 1))