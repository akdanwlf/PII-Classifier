from dotenv import load_dotenv
import requests
import os


class RecognizeWithPresidio:

    def __init__(self, text):
        pass

    @staticmethod
    def predict(input_text):  # Call the custom template for recognizing data
        analyze_endpoint = host_url() + os.getenv("ANALYZE_ENDPOINT")
        api_body = {"text": input_text,
                    "AnalyzeTemplateId": "pii_custom_template"}
        try:
            response = requests.post(analyze_endpoint, json=api_body)
            if response.status_code != 200:  # Wrong api call
                raise Exception(f"Unsuccessful API call: {response.raise_for_status()}")
            if response.json() is None:
                return None
            else:
                return recognize_by_priority(input_text, response)
        except requests.exceptions.RequestException as e:  # Failed API call
            raise SystemExit(e)


def get_output(input_text, label, start, end):  # construct output for custom usage
    label_mapping = {
        'CREDIT_CARD': 'CreditCardNumber',
        'PERSON': 'Name',
        'EMAIL': 'Email',
        'PHONE_NUMBER': 'Phone_number',
        'PLATES': 'Plates',
        'US_SSN': 'SSN'
    }
    return {
        'label': label_mapping[label],
        'PII': input_text[start:end]
    }


def recognize_by_priority(input_text, response_data):  # Apply priority on top of recognitions
    max_score = -1
    final_index = []
    for idx in range(0, len(response_data.json())):
        recognizer_result = response_data.json()[idx]
        if recognizer_result['field']['name'] == 'US_SSN':  # SSN is given top priority
            final_index.append(idx)
            break
        elif (recognizer_result['field']['name'] == 'EMAIL_ADDRESS'
              or recognizer_result['field']['name'] == 'DOMAIN_NAME'):  # If domain name or email address found
            final_index.append(idx)
            if len(final_index) > 2:
                break
        max_score = max(max_score, recognizer_result['score'])  # rest of the fields

    if len(final_index) == 1:
        result = response_data.json()[final_index[0]]
        location = result['location']
        start = location['end'] - location['length']
        return get_output(input_text, 'US_SSN', start, location['end'])
    elif len(final_index) > 1:
        result = response_data.json()
        score = (result[0]['score'] + result[1]['score']) / 2
        start_index = 0 if result[0]['field']['name'] == 'EMAIL_ADDRESS' else 1
        end_index = 0 if start_index == 0 else 1
        start = result[start_index]['location']['end'] - result[start_index]['location']['length']
        end = result[end_index]['location']['end']
        return get_output(input_text, 'EMAIL', start, end, )
    else:
        for r in response_data.json():
            if r['score'] == max_score:
                entity = r['field']['name']
                location = r['location']
                return get_output(input_text, entity, location['end'] - location['length'], location['end'])


def host_url():  # Returns Presidio API Host URL
    load_dotenv()
    return os.getenv('HOST') + ':' + os.getenv('PORT') + os.getenv('BASE_URL')
