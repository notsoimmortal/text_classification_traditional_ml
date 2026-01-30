import pickle
from preprocess import clean_text
from config import MODEL_PATH


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def predict(text: str):
    text = clean_text(text)
    return model.predict([text])[0]


if __name__ == '__main__':
    while True:
        text = input("Enter text: ")
        if text == 'exit':
            break
        print("Prediction:", predict(text))