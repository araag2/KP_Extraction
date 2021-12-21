import csv
import os, os.path
import re


import metapub
from typing import List
from utils.IO import read_from_file, write_to_file

def map_orig_links(dir_input : str, dir_output : str) -> None:
    with open(dir_input, 'r', encoding='utf-8') as file:
        id_types = ["DOI", "PMID", "ISSN", "Web", "Other"]
        results = {id:set() for id in id_types}

        regex = {"DOI" : re.compile("^10.\d{4,9}[/=][-._;()/:a-zA-Z0-9]+"), 
                 "PMID" : re.compile("[0-9]{6,8}"), 
                 "ISSN" : re.compile("ISSN:.*"), 
                 "Web" : re.compile("https?://.*"), 
                 "Other" : True}

        for line in [line.rstrip() for line in file.readlines()]:
            for pat in regex:
                if regex[pat] == True or regex[pat].match(line):
                    results[pat].add(line)
                    break
        
        print("Extraction statistics")
        for type in results:
            print(f'{type} = {len(results[type])} entries')

    print(f'Outputing to {os.getcwd()}{dir_output}')
    write_to_file(f'{os.getcwd()}{dir_output}', results)
    print('Success!')

def extract_files_from_link(dir_input : str, dir_output : str) -> None:
    doc_ids = read_from_file(dir_input)
    for type in doc_ids:
        print(f'{type} = {len(doc_ids[type])} entries')

    pmid_fetcher = metapub.PubMedFetcher()
    for entry in doc_ids["PMID"]:
        src = metapub.FindIt(entry)
        print(src.url)

    #csvreader = csv.reader(file)
    #doi_ids = set()

    #for row in csvreader:
    #    if row[0] not in doi_ids:
    #        doi_ids.add(row[0])
    #        #if row[0][0:4] == "http":
    #        #    print(row[0] + '\n')

    #with open(dest_path, 'w') as output:
    #    for key in sorted(doi_ids):
    #        try:
    #            output.write(f'{key}\n')

    #        except UnicodeEncodeError:
    #            print(key)



csv_path = "\\raw_data\\ResisBank\\src\\ResisBank_bank.csv"
raw_path = "\\raw_data\\ResisBank\\src\\ResisBank_dict_ID_dump"
dict_path = "\\raw_data\\ResisBank\\src\\ResisBank_dict_ID_dump"
out_path = "\\raw_data\\ResisBank\\src\\ResisBank_PMID_ID_dump"
pdf_path = "\\raw_data\\ResisBank\\src\\pdfs\\PMID\\"

extract_files_from_link(f'{os. getcwd()}{dict_path}', f'{os. getcwd()}{out_path}')
#map_orig_links(f'{os. getcwd()}{txt_path}', "\\raw_data\\ResisBank\\src\\ResisBank_dict_ID_dump.txt")