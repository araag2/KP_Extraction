import csv
import os, os.path

from typing import List
from utils.IO import read_from_file, write_to_file

def extract_csv_to_file(og_path : str = "", dest_path : str = ""):
    with open(og_path, 'r', encoding='utf-8') as file:
        csvreader = csv.reader(file)
        doi_ids = set()

        for row in csvreader:
            if row[0] not in doi_ids:
                doi_ids.add(row[0])
                #if row[0][0:4] == "http":
                #    print(row[0] + '\n')

        with open(dest_path, 'w') as output:
            for key in sorted(doi_ids):
                try:
                    output.write(f'{key}\n')

                except UnicodeEncodeError:
                    print(key)


og_path = "C:/Users/artur/Desktop/stuff/IST/Thesis/Code/KP_Extraction/datasets/raw_data/DOI/src/DOI_bank.csv"
dest_path = "C:/Users/artur/Desktop/stuff/IST/Thesis/Code/KP_Extraction/datasets/raw_data/DOI/src/DOI_raw_ID_dump.txt"
pdf_path = "C:/Users/artur/Desktop/stuff/IST/Thesis/Code/KP_Extraction/datasets/raw_data/DOI/pdfs/"

extract_csv_to_file(og_path, dest_path)