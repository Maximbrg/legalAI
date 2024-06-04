import logging
import os
import sys
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, pred_sentencing_path)

from model_handler import Classifier

def models_name_extraction(classifiers_path:str = None, level:str = None):
    models_name = []
    classifiers_path = os.path.join(classifiers_path, f'{level}_level')
    for model_name in os.listdir(classifiers_path):
        models_name.append(model_name)
    return models_name

def load_all_classifies(classifiers_path:str = None, first_level_labels:list = None,
                        second_level_labels:list = None, logger:logging.Logger = None):
    """
    Loads all classifiers (both first-level and second-level) from the specified model paths.
    Returns:
        dict: A dictionary containing all loaded classifiers.
        ** if one of the label list (first/second) is None -> load from folder models the models name and 
           return this lists
    """
    
    classifiers = {}
    return_labels = False
    path = os.path.join(classifiers_path, "first_level", "REJECT")
    path = os.path.join(path, load_models_name(path))
    classifiers['REJECT'] = Classifier(path, "REJECT").load_model(logger)
    
    if first_level_labels is None:
        first_level_labels = models_name_extraction(classifiers_path=classifiers_path,
                                                    level='first')
        return_labels = True
        
    for label in first_level_labels:
        if label.lower() == "reject":
            continue
        path = os.path.join(classifiers_path, "first_level", f"{label}")
        path = os.path.join(path, load_models_name(path))
        classifiers[label] = Classifier(path, label).load_model(logger)

    if second_level_labels is None:
        second_level_labels = models_name_extraction(classifiers_path=classifiers_path,
                                                     level='second')
        return_labels = True
        
    for label in second_level_labels:
        path = os.path.join(classifiers_path, "second_level", f"{label}")
        path = os.path.join(path, load_models_name(path))
        classifiers[label] = Classifier(path, label).load_model(logger)
        
    if return_labels:
        return classifiers, first_level_labels, second_level_labels
    return classifiers


def load_models_name(path):
    """
    Helper function to get the name of the model from a given directory.
    Args:
        path (str): The directory path where the models are stored.
    Returns:
        str: The name of the model.
    """

    list_dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return list_dir[0]