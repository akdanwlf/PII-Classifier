from dotenv import load_dotenv
import requests
import os
load_dotenv()

HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
BASE_URL = os.getenv('BASE_URL')

def host_url():
    return HOST + ':' + PORT + BASE_URL


def pii_custom_template():
    create_template_endpoint = host_url() + os.getenv("CREATE_TEMPLATE_ENDPOINT")
    field_names = ['CREDIT_CARD', 'DOMAIN_NAME', 'EMAIL_ADDRESS',
                   'PERSON', 'PHONE_NUMBER', 'US_SSN', 'PLATES']
    fields = list(map(lambda x: {'name': x}, field_names))
    input_body = {
        "allFields": False,
        "fields": fields
    }
    response = requests.post(create_template_endpoint, json=input_body)
    print(f'{response.status_code}:{response.json()}')


def create_recognizer(recognizer_id, entity, patterns, context_phrases=None):
    create_recogniser_endpoint = host_url() + os.getenv("NEW_RECOGNIZER_ENDPOINT") + "/" + recognizer_id
    input_body = {
        "value": {
            "entity": entity,
            "language": "en",
            "name": recognizer_id,
            "patterns": patterns
        }
    }
    if context_phrases is not None:
        input_body['context_phrases'] = context_phrases
    response = requests.post(create_recogniser_endpoint, json=input_body)
    print(f'{response.status_code}:{response.json()}')


def define_car_plates_recognizer():
    entity = 'PLATES'
    context_phrases = ['vehicle registration plate', 'vehicle plate',
                       'car plate', 'license plate',
                       'plate', 'tag', 'license']
    recognizer_id = 'plates_recognizer'
    patterns = [{
        'name': 'plates_custom_pattern',
        'regex': r"\b(([A-Z0-9]{1,4}[[\s\-]?]?[0-9A-Z]{1,5})\b|([0-9]{1,4}[\s\-]?[A-Z]{1,5)\b)",
        'score': 0.2
    }]
    create_recognizer(recognizer_id, entity, patterns, context_phrases)


def add_custom_phone_number_recognizer():
    entity = 'PHONE_NUMBER'
    recognizer_id = 'phone_number_recognizer_updated'
    patterns = [{
        'name': 'us_country_code_with_extension_strong',
        'regex': r"(((001)?|(\+1)?)[-\.\s]??\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}x?\d*|((001)?|(\+1)?)[-\.\s]??d{3}[-\.\s]\d{3}[-\.\s]\\d{4}x?\d*)",
        'score': 0.8
    }, {
        'name': 'us_country_code_with_extension_medium',
        'regex': r"(((001)?|(\+1)?)[-\.\s]??\d{3}[-\.\s]\d{3}[-\.\s]??\d{4}x?\d*)",
        'score': 0.7
    }]
    create_recognizer(recognizer_id, entity, patterns)


define_car_plates_recognizer()
add_custom_phone_number_recognizer()
pii_custom_template()
