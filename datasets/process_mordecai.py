import json
from mordecai import Geoparser

def process_MKDUC01():
	parser = Geoparser()

	with open('/home/aguimaraes/Thesis/KP_Extraction/datasets/raw_data/MKDUC-01/MKDUC01.json', 'r') as s_json:
		docs = json.load(s_json)
		res = {}

		for doc_group in docs:
			res[doc_group] = {}
			for doc in docs[doc_group]["documents"]:
				res[doc_group][doc] = str(parser.geoparse(docs[doc_group]["documents"][doc]))

	with open('/home/aguimaraes/Thesis/KP_Extraction/datasets/raw_data/MKDUC-01/MKDUC01-mordecai.json', 'w') as d_json:
		json.dump(res, d_json, indent=4, separators=(',', ': '))

process_MKDUC01()
