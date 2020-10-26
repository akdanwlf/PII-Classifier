from dotenv import load_dotenv
import pandas as pd
import spacy
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns


class BuildNERSpacyModel:

    def __init__(self):
        load_dotenv()

        input_path = os.path.join(os.getcwd(), '..', 'data', os.getenv('TRAINING_DATA_FILE'))
        print(input_path)
        train_df, test_df = self.read_data(input_path)  # get the right input
        print(train_df['Labels'].unique())
        train_data_spacy_ready = self.prepare_for_spacy(train_df, 'Text', 'PII')
        address_nlp = self.train_spacy(train_data_spacy_ready, spacy.blank("en"), 12)  # get custom entity ner model

        test_df.to_excel(os.path.join(os.getcwd(), '..', 'data', 'test.xlsx'), index=False)

        model_save_path = os.path.join(os.getcwd(), '..', 'custom_NER_model')
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
            lossArray = []
            itns = []
            for itn in range(iterations):
                print("Starting iteration... " + str(itn))
                random.shuffle(train_data)
                losses = {}

                for text, annotations in train_data:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)  # Calculates difference between the training example and the expected output
                print(losses)
                lossArray.append(losses['ner'])
                itns.append(itn + 1)

        sns.lineplot(data=pd.DataFrame({'iterations': itns, 'loss': lossArray}), x='iterations', y='loss')
        plt.savefig(os.path.join(os.getcwd(), '..', 'data', 'lossvsiterations.png'))
        return nlp

    @staticmethod
    def prepare_for_spacy(df, text_field, label_field):  # prepare the data for spacy model consumption
        data = []
        for idx, row in df.iterrows():
            label_start_index = row[text_field].find(row[label_field])
            label_end_index = label_start_index + len(row[label_field])
            if row[text_field] != '' and row['Labels'] != 'None' and label_start_index != -1:
                ent = {'entities': [(label_start_index, label_end_index, row['Labels'])]}
                data.append((row[text_field], ent))  # each input should contain the text and entities with annotations
            else:
                print(f"skipping {idx} {row['Labels']}")
        return data

    @staticmethod
    def read_data(path):  # read the file, remove unused data and strip spaces
        df = pd.ExcelFile(path)
        df = pd.read_excel(df, os.getenv('TRAINING_DATA_SHEET'), skiprows=1)
        print(f'Reading file {path}')
        df = df[df['Labels'] != 'Plates']
        df = df.astype('str')
        for col in df.columns:
            df[col] = df[col].str.strip()
        train = df.sample(random_state=322, frac=0.8)
        test = df.drop(train.index)
        return train, test


if __name__ == '__main__':
    BuildNERSpacyModel()
