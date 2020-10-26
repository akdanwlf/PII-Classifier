import os
import pandas as pd
import spacy

test_file_path = os.path.join(os.getcwd(), '..', 'data', 'test.xlsx')
df = pd.ExcelFile(test_file_path)
df = pd.read_excel(df, 'Sheet1')
df = df.astype('str')
for col in df.columns:
    df[col] = df[col].str.strip()

ner_new = spacy.load(os.path.join('..', 'custom_NER_model'))
for row in df.itertuples():
    doc = ner_new(row.Text)
    df.loc[row.Index, 'prediction_label'] = '' if len(doc.ents) < 1 else doc.ents[0].label_
    df.loc[row.Index, 'prediction_PII'] = 'None' if len(doc.ents) < 1 else doc.ents[0].text

file_name = ['predictions', '0', '.xlsx']
if not os.path.isdir('data'):
    os.mkdir('data')
output_file_path = os.path.join(os.getcwd(), '..', 'data', ''.join(file_name))
while os.path.isfile(output_file_path):
    file_name[1] = str(int(file_name[1]) + 1)
    output_file_path = os.path.join(os.getcwd(), '..', 'data', ''.join(file_name))
print(f"outputs saved to {file_name}")
df.to_excel(output_file_path, index=False)
