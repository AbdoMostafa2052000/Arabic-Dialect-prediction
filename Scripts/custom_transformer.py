# custom_transformer.py

from sklearn.base import BaseEstimator, TransformerMixin
import re
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
        # Remove non-Arabic characters
        arabic_text = re.sub(r'[^\u0600-\u06FF\s]+', '', text)
        
        # Remove URLs and mentions
        arabic_text = re.sub(r'http\S+|@\w+', '', arabic_text)
        
        # Remove Arabic stopwords
        arabic_text = ' '.join(word for word in arabic_text.split() if word not in stopwords_arabic)
        
        # Optionally, remove additional characters or patterns specific to your use case
        
        return arabic_text.strip()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
