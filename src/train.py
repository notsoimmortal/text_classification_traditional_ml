import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from data_loader import load_data
from preprocess import clean_text
from features import build_vectorizer
from config import *
from utils import setup_logger

logger = setup_logger()

def main():
    os.makedirs("../data/processed", exist_ok=True)

    if os.path.exists(PROCESSED_PATH):
        logger.info("Loading preprocessed dataset")
        df = load_data(PROCESSED_PATH)
    else:
        logger.info("Loading raw dataset")
        df = load_data(DATA_PATH)

        logger.info("Cleaning text")
        df['text'] = df['text'].apply(clean_text)

        df.to_csv(PROCESSED_PATH, index=False)
        logger.info(f"Preprocessed data saved to {PROCESSED_PATH}")

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', build_vectorizer()),
        ('clf', LinearSVC(class_weight='balanced', max_iter=10000))
    ])

    logger.info("Training model")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    logger.info("Saving model")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    logger.info("Training complete")

if __name__ == '__main__':
    main()
