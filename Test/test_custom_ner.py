import os
import pandas as pd
import spacy


def predict_on_test_data():
    test_file_path = os.path.join(os.getcwd(), '..', 'data', 'test.xlsx')
    df = pd.ExcelFile(test_file_path)
    df = pd.read_excel(df, 'Sheet1')
    df = df.astype('str')
    for col in df.columns:
        df[col] = df[col].str.strip()

    for row in df.itertuples():
        doc = get_prediction(row.Text)
        df.loc[row.Index, 'prediction_label'] = 'None' if len(doc.ents) < 1 else doc.ents[0].label_
        df.loc[row.Index, 'prediction_PII'] = '' if len(doc.ents) < 1 else doc.ents[0].text
        print(row.Index, ' ', df.loc[row.Index, 'prediction_label'], ' ', df.loc[row.Index, 'prediction_PII'])

    file_name = ['ner_model_predictions', '0', '.xlsx']
    if not os.path.isdir(os.path.join(os.getcwd(), '..', 'data')):
        os.mkdir(os.path.join(os.getcwd(), '..', 'data'))
    output_file_path = os.path.join(os.getcwd(), '..', 'data', ''.join(file_name))
    while os.path.isfile(output_file_path):
        file_name[1] = str(int(file_name[1]) + 1)
        output_file_path = os.path.join(os.getcwd(), '..', 'data', ''.join(file_name))
    print(f"outputs saved to {file_name}")
    df.to_excel(output_file_path, index=False)


def get_prediction(text):
    ner_new = spacy.load(os.path.join('..', 'custom_NER_model'))
    doc = ner_new(text)
    return doc


if __name__ == '__main__':
    text = input("Enter the text or press enter to predict on test data file\n")
    if not text:
        predict_on_test_data()
    else:
        doc = get_prediction(text)
        print('None' if len(doc.ents) < 1 else doc.ents[0].label_)
        print('' if len(doc.ents) < 1 else doc.ents[0].text)
