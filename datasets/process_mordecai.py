import json
from mordecai import Geoparser

def process_MKDUC01():
    parser = Geoparser()

    with open(f'..\\raw_data\\MKDUC-01\\MKDUC01.json', 'r') as s_json:
        docs = json.load(s_json)

        print(docs)