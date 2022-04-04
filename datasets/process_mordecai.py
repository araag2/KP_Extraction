import json
from mordecai import Geoparser

def process_MKDUC01():
    parser = Geoparser()

    with open('..\\raw_data\\MKDUC-01\\MKDUC01.json', 'r') as s_json:
        docs = json.load(s_json)

        print(docs)

        with open('..\\raw_data\\MKDUC-01\\MKDUC01-Mordecai.json', 'w') as d_json:
            json.dump(docs, d_json, indent=4, separators=(',', ': '))