from sklearn.feature_extraction.text import TfidfVectorizer
from config import MAX_FEATURES, NGRAM_RANGE


def build_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=1,
        # max_df=0.9,
        sublinear_tf=True
    )