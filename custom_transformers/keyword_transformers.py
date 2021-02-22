import re

from sklearn.base import BaseEstimator, TransformerMixin
from nltk import WordNetLemmatizer

class CleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words):
        self._stop_words = stop_words
        pass

    def fit(self):
        pass

    def transform(self, hl):
        # Remove punctuation
        hl = re.sub('[^a-zA-Z]', ' ', str(hl))

        # Convert to lowercase
        hl = hl.lower()

        # Remove tags
        hl = re.sub('', '', hl)
        hl = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", hl)

        # Remove special characters and digits
        hl = re.sub("(\\d|\\W)+"," ", hl)

        # Convert to list from string
        hl = hl.split()

        # Stemming (also does lowercase and removes punctuation)
        # ps = PorterStemmer()
        # hl = [ps.stem(x) for x in hl]

        # Lemmatisation
        lem = WordNetLemmatizer()
        hl = [lem.lemmatize(word) for word in hl
              if not word in self._stop_words]
        return [' '.join(hl)]

class ScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cv):
        self._cv = cv
        pass

    def fit(self):
        pass

    def predict(self, X, n=5):
        sorted_items = self._sort_coo(X.tocoo())
        return self._extract_topn_from_vector(self._cv.get_feature_names(), sorted_items, n)

    def _sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        feature_vals = [feature_names[idx] for idx, _ in sorted_items]
        score_vals = [round(score, 3) for _, score in sorted_items]

        return feature_vals, score_vals
