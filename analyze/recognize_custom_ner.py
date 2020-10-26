import spacy
import os


class RecognizeWithCustomNER:

    address_model = ''

    def __init__(self):
        pass

    @staticmethod
    def predict(text):  # predict address using Spacy Address model
        address_model = load_model()
        doc = address_model(text)
        label = None
        address = None
        len(doc.ents)
        if len(doc.ents) >= 1:
            address = doc.ents[0].text
            label = doc.ents[0].label_
        else:
            return None
        return {"label": label, "PII": address}


def load_model():
    #   loading model from folder address_model in current working directory
    address_model_path = os.path.join(os.getcwd(), 'custom_NER_model')
    if os.path.isdir(address_model_path):
        return spacy.load(address_model_path)
    else:
        raise IOError(f"Address model not found at {address_model_path}")