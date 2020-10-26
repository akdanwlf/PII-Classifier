from dotenv import load_dotenv
import pandas as pd
import spacy
import random
import os


class BuildAddressSpacyModel:

    def __init__(self):
        load_dotenv()

        input_path = os.path.join(os.getcwd(), '..', 'data', os.getenv('TRAINING_DATA_FILE'))
        print(input_path)
        train_df = self.read_data(input_path)  # get the right input
        train_data_spacy_ready = self.prepare_for_spacy(train_df, 'Text', 'PII', 'LOC')
        address_nlp = self.train_spacy(train_data_spacy_ready, spacy.load("en_core_web_sm"), 10)  # get custom entity ner model

        model_save_path = os.path.join(os.getcwd(), '..', 'address_model1')
        print(f"model is saved at {model_save_path}")
        address_nlp.to_disk(model_save_path)  # Save Address Model

    @staticmethod
    def train_spacy(train_data, model, iterations):
        nlp = model  # create blank Language class

        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)
        else:
            ner = nlp.get_pipe('ner')

        # add labels
        for _, annotations in train_data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(iterations):
                print("Starting iteration... " + str(itn))
                random.shuffle(train_data)
                losses = {}
                for text, annotations in train_data:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.3,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)  # Calculates difference between the training example and the expected output
                print(losses)

        return nlp

    @staticmethod
    def prepare_for_spacy(df, text_field, label_field, label):  # prepare the data for spacy model consumption
        data = []
        for idx, row in df.iterrows():
            label_start_index = row[text_field].find(row[label_field])
            label_end_index = label_start_index + len(row[label_field])
            if row[text_field] != '' and label_start_index != -1:
                ent = {'entities': [(label_start_index, label_end_index, label)]}
                data.append((row[text_field], ent))  # each input should contain the text and entities with annotations

        return data

    @staticmethod
    def read_data(path):  # read the file, remove unused data and strip spaces
        df = pd.ExcelFile(path)
        df = pd.read_excel(df, os.getenv('TRAINING_DATA_SHEET'), skiprows=1)
        print(f'Reading file {path}')
        df.drop(df[df['Labels'] != 'Address'].index, inplace=True)
        df.reset_index(drop=True)
        df = df[['Text', 'PII']]
        df = df.astype('str')
        for col in df.columns:
            df[col] = df[col].str.strip()
        return df


if __name__ == '__main__':
    BuildAddressSpacyModel()