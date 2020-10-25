from dotenv import load_dotenv
import pandas as pd
import requests
import os

load_dotenv()

HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
BASE_URL = os.getenv('BASE_URL')


def host_url():
    return HOST + ':' + PORT + BASE_URL


def presidio_recognize(input_text):
    analyze_endpoint = host_url() + os.getenv("ANALYZE_ENDPOINT")
    api_body = {"text": input_text,
                "AnalyzeTemplateId": "pii_custom_template"}
    response = requests.post(analyze_endpoint, json=api_body)
    if response.status_code != 200:
        return None
    if response.json() is None:
        return None
    else:
        return recognize_by_priority(response)


def recognize_by_priority(response_data):
    max_score = -1
    final_index = []
    for idx in range(0, len(response_data.json())):
        recognizer_result = response_data.json()[idx]
        if recognizer_result['field']['name'] == 'US_SSN':
            final_index.append(idx)
            break
        elif (recognizer_result['field']['name'] == 'EMAIL_ADDRESS'
              or recognizer_result['field']['name'] == 'DOMAIN_NAME'):
            final_index.append(idx)
            if len(final_index) > 2:
                break
        max_score = max(max_score, recognizer_result['score'])

    if len(final_index) == 1:
        result = response_data.json()[final_index[0]]
        start = result['location']['end'] - result['location']['length']
        return get_output('US_SSN', start, result['location']['end'], result['location']['length'], result['score'])
    elif len(final_index) > 1:
        result = response_data.json()
        score = (result[0]['score'] + result[1]['score']) / 2
        start_index = 0 if result[0]['field']['name'] == 'EMAIL_ADDRESS' else 1
        end_index = 0 if start_index == 0 else 1
        start = result[start_index]['location']['end'] - result[start_index]['location']['length']
        end = result[end_index]['location']['end']
        return get_output('EMAIL', start, end, end - start, score)
    else:
        for r in response_data.json():
            if r['score'] == max_score:
                return get_output(r['field']['name'], r['location']['end'] - r['location']['length']
                                  , r['location']['end'], r['location']['length'], r['score'])


def get_output(label, start, end, length, score):
    return {
        'label': label,
        'score': score,
        'location': {
            'start': start,
            'end': end,
            'length': length
        }
    }


def read_data(path):
    df = pd.ExcelFile(path)
    df = pd.read_excel(df, os.getenv('TRAINING_DATA_SHEET'), skiprows=1)
    df = df.astype('str')
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


if __name__ == "__main__":

    input_path = os.getenv('TRAINING_DATA_PATH')
    train_df = read_data(os.path.join(os.path.dirname(__file__), input_path))

    for row in train_df.itertuples():
        prediction = presidio_recognize(row.Text)
        train_df.loc[row.Index, 'prediction_label'] = 'None' if prediction is None else prediction['label']
        if prediction is None:
            train_df.loc[row.Index, 'prediction_PII'] = ''
        else:
            train_df.loc[row.Index, 'prediction_PII'] = row.Text[prediction['location']['start']:prediction['location']['end']]

    file_name = ['predictions', '0', '.xlsx']
    output_file_path = os.path.join(os.getcwd(), 'data', ''.join(file_name))
    while os.path.isfile(output_file_path):
        file_name[1] = str(int(file_name[1]) + 1)
        output_file_path = os.path.join(os.getcwd(), 'data', ''.join(file_name))
    train_df.to_excel(output_file_path, index=False)






