# custom_transformer.py

from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from arabic_reshaper import reshape
from nltk.corpus import stopwords

stopwords_arabic = set(stopwords.words('arabic'))


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = X.copy()
        transformed_X = transformed_X.apply(self.clean_arabic_text)
        return transformed_X

    @staticmethod
    def clean_arabic_text(text):
        # Define Arabic stopwords
        stopwords_arabic = set(stopwords.words('arabic'))
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove punctuation and numbers
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        
        # Convert to Arabic script
        text = reshape(text)
        # Remove diacritics
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        # Normalize Arabic letters
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords_arabic]
        
        # Stemming
        stemmer = ISRIStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
