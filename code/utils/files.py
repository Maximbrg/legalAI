import pickle
import csv
import json
import os
# from docx import Document
import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime
# from colorama import Fore, Style


def read_docx(docx_path):
    """
    Read the content of a DOCX file and return the entire text.

    Parameters:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The entire text content of the document.
    """
    try:
        doc = Document(docx_path)
        text_content = ''

        for paragraph in doc.paragraphs:
            text_content += paragraph.text + '\n'
            
        return text_content.strip()  # Strip trailing newline

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_pkl(path):
    with open(path, 'rb') as file:
        content = pickle.load(file)
    return content


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f)


def create_csv(file_path, headers):
    # Write data to the CSV file
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the headers
        writer.writerow(headers)



def append_to_csv(file_path, data):
    # Append data to the CSV file
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the data
        writer.writerow(data)
        

def flatten_dict(d):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[k] = v
    return items


def config_parser(config_name):
    """
    Reads a YAML file and returns its contents as a flattened dictionary.

    :param filename: Path to the YAML file.
    :return: Flattened dictionary representation of the YAML file.
    """
    config_path = os.path.join(os.path.abspath(__file__).split('src')[0],
                               'resources/configs', config_name + '.yaml')
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        return data
    
def yaml_load(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        return data

def setup_logger(save_path, file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    current_date_time = datetime.now()
    current_date = current_date_time.date()

    log_file = os.path.join(save_path,  f'{current_date}_{file_name}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add color to console output
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            log_str = super().format(record)
            if record.levelname == 'DEBUG':
                return f'{Fore.GREEN}{log_str}{Style.RESET_ALL}'
            elif record.levelname == 'WARNING':
                return f'{Fore.YELLOW}{log_str}{Style.RESET_ALL}'
            elif record.levelname == 'ERROR':
                return f'{Fore.RED}{log_str}{Style.RESET_ALL}'
            elif record.levelname == 'CRITICAL':
                return f'{Style.BRIGHT}{Fore.RED}{log_str}{Style.RESET_ALL}'
            else:
                return log_str

    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s',  datefmt='%Y-%m-%d %H:%M:%S'))

    return logger

def weap_type_extract(tagged_df):
    """
    over on each row (case) in tagged feature extraction csv,
    and aggrigate the approprate column - that contain weapon type information.
    """
    new_coulmn = []
    for _, row in tagged_df.iterrows():
        new_cell = []
        for column in tagged_df.columns:
            if "WEP_TYPE" in column:
                if not pd.isna(row[column]) and row[column] != "":
                    new_cell.append(column.replace('WEP_TYPE-', ''))
    
        new_coulmn.append(new_cell)
    tagged_df['TYPE_WEP'] = new_coulmn
    return tagged_df
                    
def write_yaml(save_path, content):
    with open(save_path, 'w') as file:
        yaml.dump(content, file)
        


def reformat_sentence_tagged_file(case_dir_path):
    """
    Convert a tagged CSV file to a new format with 'text' and 'label' columns.

    Parameters:
    - tagged_csv_path (str): The file path of the tagged CSV file.

    Returns:
    - pd.DataFrame: A new DataFrame with 'text' and 'label' columns.
    """
    sentence_tagging_path = os.path.join(case_dir_path, 'sentence_tagging.csv')
    df = pd.read_csv(sentence_tagging_path)
    data = []
    for _, row in df.iterrows():
        text = row['text']

        for column, value in row.items():
            if column.lower() != 'text' and column.lower() != 'verdict' and value == 1 and column.lower() != 'reject':
                data.append({'text': text, 'label': column})

    new_df = pd.DataFrame(data)
    return new_df