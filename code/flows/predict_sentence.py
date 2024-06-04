import logging
import os
import sys
from datetime import datetime


current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import setup_logger, config_parser
from utils.errors.predict_sentence_cls import *
from scripts.sentence_classification.predict_sentence_cls.setfit_postprocessor import SetFitPostProcessor
from scripts.sentence_classification.predict_sentence_cls.loads import load_all_classifies



def predict_2cls_lvl_flow(case_dir_path:str = None, preprocess_file_path:str = None, classifiers_path:str = None, 
                          result_path:str = None, logger:logging.Logger = None, first_level_labels:list = None,
                          second_level_labels:list = None, threshold:object = None, tagged_path: str = None,
                          eval_path:str = None):

    """
    In this function, it is possible move on case directory,
    and go through each sentence in the preprocess csv to predict the label (2 level),
    or move on excel tagged file perform an evaluation

    Args:
        case_dir_path: path of the verdict in db
        preprocess_file_path: the output of preprocess flow
        second/first_level_labels: list of labels the the user want to predict 

    Returns:
        str: The path to the saved predicted CSV file.
    """
    
    if (case_dir_path is None) and (tagged_path is None):
        case_dir_path_error(logger)
        
    if case_dir_path:
        if preprocess_file_path is None:
            preprocess_file_path = os.path.join(case_dir_path, 'preprocessing.csv')
        save_path = case_dir_path
        
    else:
        today = datetime.today()
        formatted_date = today.strftime("%d.%m")
        save_path = os.path.join(eval_path, formatted_date)
        preprocess_file_path = tagged_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    predict_proba_path = os.path.join(save_path, 'sentence_probability.csv')
    predict_label_path = os.path.join(save_path, 'sentence_tagging.csv')
    

    if classifiers_path is None:
        classifiers_path_error(logger)    
            
    if first_level_labels is None or second_level_labels is None:
        classifiers, first_level_labels, second_level_labels = load_all_classifies(classifiers_path=classifiers_path,
                                                                                   logger=logger)
    else:
        classifiers = load_all_classifies(classifiers_path=classifiers_path,
                                          first_level_labels=first_level_labels,
                                          second_level_labels=second_level_labels,
                                          logger=logger)
    
        
    predictor = SetFitPostProcessor(logger=logger,
                                    preprocess_file_path=preprocess_file_path,
                                    save_path=save_path,
                                    classifiers=classifiers,
                                    first_level_labels=first_level_labels,
                                    second_level_labels=second_level_labels,
                                    threshold=threshold,
                                    evaluation=False
                                    )
        
    predict_proba_df, predict_label_df = predictor.predict()
    predict_proba_df.to_csv(predict_proba_path, index=True)
    predict_label_df.to_csv(predict_label_path, index=True)
    logger.info(f"Predict probability saved in: {predict_proba_path}")
    logger.info(f"Lebels prediction save in: {predict_label_path}")
    return predict_proba_df, predict_label_df, classifiers



def main(param):
    logger = setup_logger(save_path=os.path.join(param['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')
    
    # predict_2cls_lvl_flow(case_dir_path=param['case_dir_path'],
    #                       preprocess_file_path=param['preprocess_file_path'],
    #                       classifiers_path=param['classifiers_path'],
    #                       result_path=param['result_path'],
    #                       logger=logger,
    #                       threshold=param['threshold']
    #                      )

    predict_2cls_lvl_flow(tagged_path='/home/maximbr/legalAI/data/sentence_classfication/sentence_tag_test.csv',
                          eval_path='results/evaluations/sentence_calssification',
                          classifiers_path=param['classifiers_path'],
                          result_path=param['result_path'],
                          logger=logger,
                          threshold=param['threshold']
                          )

if __name__ == '__main__':
    param = config_parser('/home/maximbr/legalAI/legalAI/code/configs/main_config')
    main(param)