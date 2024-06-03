import os
import sys

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.files import config_parser, setup_logger, write_yaml
from scripts.sentence_classification.setfit_handler import setfit_trainer


def run(params, logger):
    """
    Over on each model in models to train (in config), than train by the specipic params, and save them on 
    result/models/specipic experimennt and date running 
    """
    
    logger.info(f"Train start, data is taken from {params['data_path']}")
    
    for model_config in params['models_to_train']:
        
        if model_config[list(model_config.keys())[0]]:
            model_name, _ = list(model_config.items())[0]
            labels = params['labels']
            logger.info(f'Starting {model_name} training for labels: {labels}')
        
            if model_name == 'setfit':
                
                st = setfit_trainer(logger,
                                    train_path=params["data_path"],
                                    save_dir=params["save_dir"],
                                    num_samples_list=params["num_samples_list"],
                                    model_name_initial=params["model_name_initial"],
                                    load_xlsx=True,
                                    all_class=params["all_class"],
                                    batch_size=params["batch_size"],
                                    num_iteration=params["num_iteration"],
                                    labels_=params["labels"],
                                    pretrained_model=params["pretrained_model"],
                                    pretrained_model_list=params["pretrained_model_list"],
                                    result_path=params['result_path']
                                    )
                
                save_path = st.train(params['experiment_name'])
    write_yaml(os.path.join(save_path, 'config.yaml'), params)
    logger.info(f'{model_name} training finished.')


def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='preprocess_test')

    run(params, logger)
    

if __name__ == "__main__":
    params = config_parser("/home/maximbr/legalAI/legalAI/code/configs/train_sentence_cls")
    main(params)