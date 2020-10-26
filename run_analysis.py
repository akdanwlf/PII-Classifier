from dotenv import load_dotenv
from analyze import RecognizeWithCustomNER as custom_ner
from analyze import RecognizeWithPresidio as presidio
import pandas as pd
import os

load_dotenv()


def read_data(path):  # read the file, remove unused data and strip spaces
    df = pd.ExcelFile(path)
    df = pd.read_excel(df, os.getenv('TEST_DATA_SHEET'), skiprows=1)
    df = df[['Text']]
    df = df.astype('str')
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def output_file(df):
    file_name = ['predictions', '0', '.xlsx']
    if not os.path.isdir('data'):
        os.mkdir('data')
    output_file_path = os.path.join(os.getcwd(), 'data', ''.join(file_name))
    while os.path.isfile(output_file_path):
        file_name[1] = str(int(file_name[1]) + 1)
        output_file_path = os.path.join(os.getcwd(), 'data', ''.join(file_name))
    print(f"outputs saved to {output_file_path}")
    df.to_excel(output_file_path, index=False)


def get_prediction(text):
    custom_prediction = custom_ner.predict(text)
    presidio_prediction = presidio.predict(text)

    if custom_prediction is not None and presidio_prediction is not None:
        if custom_prediction['label'] == presidio_prediction['label']:
            print("custom_prediction")
            return custom_prediction
        elif presidio_prediction['label'] != 'Plates' and presidio_prediction['label'] != 'Name':
            print("presidio_prediction")
            return presidio_prediction
        elif custom_prediction['label'] == 'Address':
            print("custom_prediction")
            return custom_prediction
        else:
            print("presidio_prediction")
            return presidio_prediction
            # print("custom_prediction")
            # return custom_prediction
    elif custom_prediction is not None:
        print("custom_prediction")
        return custom_prediction
    elif presidio_prediction is not None:
        print("presidio_prediction")
        return presidio_prediction
    else:
        return None


if __name__ == '__main__':

    input_path = os.path.join(os.getcwd(), 'data', os.getenv('TEST_DATA_FILE'))
    dataframe = read_data(input_path)  # get the right input

    for row in dataframe.itertuples():
        prediction = get_prediction(row.Text)
        dataframe.loc[row.Index, 'prediction_label'] = 'None' if prediction is None else prediction['label']
        dataframe.loc[row.Index, 'prediction_PII'] = '' if prediction is None else prediction['PII']
        print(row.Index, ' ', prediction)

    output_file(dataframe)
