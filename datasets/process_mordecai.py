import json
import plotly.express as px
#from mordecai import Geoparser


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

def build_map():
	with open('C:\\Users\\artur\\Desktop\\stuff\\IST\\Thesis\\Code\\KP_Extraction\\datasets\\raw_data\\MKDUC-01\\MKDUC01-mordecai.json', 'r') as s_json:
		source = json.load(s_json)
		res = {"geo_loc" : [], "country_loc" : []}
		geo_locations = []
		country_locations = []

		for t in source:
			for d in source[t]:
				data_list = eval(source[t][d])
				for entry in data_list:
					if "geo" in entry:
						res["geo_loc"].append((float(entry["geo"]["lat"]), float(entry["geo"]["lon"])))
					else:
						res["country_loc"].append(entry["country_predicted"])

		with open('C:\\Users\\artur\\Desktop\\stuff\\IST\\Thesis\\Code\\KP_Extraction\\datasets\\raw_data\\MKDUC-01\\MKDUC01-geo_locations.json', 'w') as d_json:
			json.dump(res, d_json, indent=4, separators=(',', ': '))

#process_MKDUC01()
build_map()