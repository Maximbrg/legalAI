import sys
import pandas as pd
import json
import os


def convert_label2binary(df, label_target, labels: list = None):
    new_df = df[df['label'] == label_target]
    target_label_rows = new_df.shape[0]
    number_sample = target_label_rows // (len(labels) - 1)
    for label in labels:
        if label != label_target:
            num_rows = df[df['label'] == label].shape[0]
            temp = df[df['label'] == label].sample(n=min(number_sample, num_rows), random_state=42)
            new_df = pd.concat([new_df, temp])

    new_df['label'] = (new_df['label'] == label_target).astype(int)
    return new_df

def convert_all_labels2binary(df: pd.DataFrame = None, labels: list = None):
    # create separate dataframes for each label
    dfs = {}
    for label in labels:
        dfs[label] = convert_label2binary(df, label, labels=labels)
    return dfs

def jsonTag2dict(tag_file_path, encoding: str = 'utf-16'):
    """
    Parse JSON tagged data files and convert them into a Python dictionary.

    Parameters:
        tag_file_path (str): The file path for the directory containing the JSON tagged data files.

    Returns:
        dict: A dictionary where each key is a choice and the value is a list of sentences associated with that choice.
              The 'reject' label is included as a key in the dictionary, with its corresponding list of rejected sentences.
  """
    sentences_choice = {}
    choise_sentences = {'reject': []}
    for filename in os.listdir(tag_file_path):
        if filename == '.ipynb_checkpoints': continue
        with open(os.path.join(tag_file_path, filename), 'r', encoding=encoding) as file:
            # Loop over the lines in the file
            for line in file:
                # Load the JSON data from the line
                line_data = json.loads(line)

                if line_data['answer'] == 'reject':
                    if line_data['text'] not in sentences_choice:
                        sentences_choice[line_data['text']] = ['reject']
                    choise_sentences['reject'].append(line_data['text'])

                for choice in line_data['accept']:

                    if line_data['text'] not in sentences_choice:
                        sentences_choice[line_data['text']] = [choice]
                    else:
                        # if choice not in line_data['text']:
                        sentences_choice[line_data['text']].append(choice)

                    if choice not in choise_sentences:
                        choise_sentences[choice] = [line_data['text']]
                    else:
                        if choice not in line_data['text']:
                            choise_sentences[choice].append(line_data['text'])
    multy_label_dict = sentences_choice
    return choise_sentences, multy_label_dict


def dict2df(choise_sentences, SEED):
    list_df = []
    for key in choise_sentences.keys():
        if key not in ['EXTRADITION', 'REMORSE']:
            for sentence in choise_sentences[key]:
                list_df.append((sentence, key))

    df = pd.DataFrame(list_df, columns=['text', 'label'], ).sample(frac=1, random_state=SEED)
    df['label'] = df['label'].replace('UNRELATED_OFFENSE', 'OFFENSE')
    df['label'] = df['label'].replace('UNRELATED_PUNISHMENT', 'PUNISHMENT')
    return df

def preprocessing_flow(path, SEED=42, csv=None, similarity_sampling=False, under_sampling=False, load_xlsx=False,
                       labels: list = None):
    if load_xlsx:
        df = pd.read_csv(path)
        multy_label_dict = {}
    else:
        choise_sentences, multy_label_dict = jsonTag2dict(path)
        df = dict2df(choise_sentences, SEED)
    if under_sampling:
        df = under_sampling(df)
    dfs = convert_all_labels2binary(df, labels=labels)
    # print(dfs['GENERAL_CIRCUM'])
    return df, dfs, multy_label_dict


def create_multilabel_df(multi_label_dict: dict = None, labels: list = ['OFFENSE', 'PUNISHMENT', 'CONFESSION', 'reject', 'CIRCUM_OFFENSE', 'GENERAL_CIRCUM']):
    """

    :param multi_label_dict:
    :param labels:
    :return:
    """

    data = {'text': []}
    for label in labels:
        data[label] = []

    for item in multi_label_dict:

        data['text'].append(item)
        for label in labels:

            if label in multi_label_dict[item]:
                data[label].append(1)
            else:
                data[label].append(0)

    return pd.DataFrame(data)