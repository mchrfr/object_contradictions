import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Navigate to the project root

# Add utils folder to the Python path
utils_path = os.path.join(project_dir, 'utils')
sys.path.append(utils_path)

#import random
from tqdm import tqdm
#import re
#from typing import Optional

import torch
import spacy
import stanza
import datasets

import pandas as pd
from thefuzz import fuzz

from preparations import modify_sectionName, group_dataset_by_section
from processing import process_downloaded_datasets


"""output file"""

# Define relative paths for output files
output_dir = os.path.join(project_dir, 'generated_datasets')

samples_csv = os.path.join(output_dir, 'test.csv') 
#entities_csv = os.path.join(output_dir, 'test_ent.csv')  # nur bei name adding notwendig

samples_json = os.path.join(output_dir, 'json', 'test.json')
#entities_json = os.path.join(output_dir, 'json', 'test_ent.json')  # nur bei name adding notwendig


if torch.cuda.is_available():
    device = torch.device("cuda:3")
    print("GPU available")
else:
    device = torch.device("cpu")
    print("using CPU ...")


if spacy.prefer_gpu():
    print("GPU will be used with spaCy.")
else:
    print("GPU will not be used with spaCy.")  # ToDo: Warum wird keine GPU für spacy verwendet?

spacy_model = spacy.load("en_core_web_trf")
print(type(spacy_model))

stanza.download('en')
stanza_model = stanza.Pipeline('en', use_gpu=True)


if __name__ == "__main__":

    # download datasets
    march = datasets.load_dataset("RealTimeData/bbc_news_march_2023")
    april = datasets.load_dataset("RealTimeData/bbc_news_april_2023")
    may = datasets.load_dataset("RealTimeData/bbc_news_may_2023")
    june = datasets.load_dataset("RealTimeData/bbc_news_june_2023")
    july = datasets.load_dataset("RealTimeData/bbc_news_july_2023")

    month = [march, april, may, june, july]

    # combine datasets and create a pandas dataframe
    combined_dataset = datasets.concatenate_datasets([ds['train'] for ds in month])
    combined_dataset.set_format('pandas')
    df = combined_dataset[:]
    data = pd.DataFrame([df.description, df.section,]).transpose()

    # reorganize sections names
    data.section = data.section.apply(modify_sectionName)
    sectionFrames_list = group_dataset_by_section(data)

    # Import named entities categories 
    entitiesHub_path = os.path.join(project_dir, 'file_sources', 'entitiesHub_with_names.txt')
    entitiesHub = pd.read_csv(entitiesHub_path, delimiter='\t')

    # Create contradictory samples
    samples = process_downloaded_datasets(tqdm(sectionFrames_list), 
                                           entitiesHub, 
                                           spacy_model, 
                                           stanza_model)

    try:
        os.makedirs(os.path.dirname(samples_csv), exist_ok=True)
        samples.to_csv(samples_csv, sep=';')
        # entities.to_csv(entities_csv, sep=';')  # nur notwendig bei name Addings

        print(f"CSV files have been created.")

    except Exception as e:
        print(f"{e}.\nCould not create CSV files.")
        
    try:
        os.makedirs(os.path.dirname(samples_json), exist_ok=True)
        samples.to_json(samples_json, orient='records', default_handler=str)

    except Exception as e:
        print(f"{e}.\nCould not create JSON files.")

    print(f"{len(samples)} samples have been created (contradictory and not contradictory). The API calls amount to a total cost of {samples.total_cost.sum()}")