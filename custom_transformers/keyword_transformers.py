import re

from sklearn.base import BaseEstimator, TransformerMixin

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
        #hl = hl.split()

        # Stemming (also does lowercase and removes punctuation)
        #ps = PorterStemmer()
        #hl = [ps.stem(x) for x in hl]

        # Lemmatisation
        #lem = WordNetLemmatizer()
        #hl = [lem.lemmatize(word) for word in hl 
        #      if not word in self._stop_words] 
        return [hl]
    
class ScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cv):
        self._cv = cv
        pass
    
    def fit(self):
        pass
    
    def predict(self, X):
        sorted_items = self._sort_coo(X.tocoo())
        return self._extract_topn_from_vector(self._cv.get_feature_names(), sorted_items, 10)
    
    def _sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:

            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        #create a tuples of feature,score
        results = zip(feature_vals,score_vals)
        #results= []
        #for idx in range(len(feature_vals)):
        #    results.append((feature_vals[idx], score_vals[idx]))

        return list(results)