import pandas as pd
import datetime
import torch
import sys
import os
from setfit import SetFitModel, SetFitTrainer

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.sentence_classification import evaluate, save_model, create_setfit_logger, load_datasets
from scripts.sentence_classification.data_handler import data_handler

dfs = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class setfit_trainer:
    def __init__(self, logger, train_path: str = None,
                 save_dir: str = None,
                 num_samples_list: int = None,
                 model_name_initial: str = None,
                 all_class: bool = None,
                 batch_size: int = None,
                 num_iteration: int = None,
                 labels_: list = None,
                 positive_number: int = 8,
                 pretrained_model: str = None,
                 result_path: str = None,
                 load_xlsx: bool = None,
                 SEED: int = 7,
                 pretrained_model_list: str = None):
        """
        Initialize the SetFit trainer.

        This class constructor initializes the SetFit trainer and its parameters for model training.
        The trainer is responsible for handling the data, setting hyperparameters, and training the models.

        Parameters:
            train_path (str, optional): The file path for the training data. Defaults to None.
            save_dir (str, optional): The directory path to save the trained models and results. Defaults to None.
            num_samples_list (int, optional): The number of samples in the dataset. Defaults to None.
            model_name_initial (str, optional): The initial model name to be used for saving checkpoints during training. Defaults to None.
            param (dict, optional): A dictionary containing hyperparameters for the models. Defaults to None.
            all_class (bool, optional): A flag indicating whether to consider all classes during model training. Defaults to None.
            batch_size (int, optional): The batch size for training. Defaults to None.
            num_iteration (int, optional): The number of iterations for training. Defaults to None.
            epoch (int, optional): The number of epochs for training. Defaults to None.
            labels_ (list, optional): A list of labels for the classes to be used during training. Defaults to None.
            positive_number (int, optional): The number of positive instances for each label during training. Defaults to 8.
            pretrained_model (str, optional): The path to a pre-trained model for transfer learning. Defaults to None.
            load_xlsx (bool, optional): A flag indicating whether to load data from an Excel file. Defaults to None.
            SEED (int, optional): The random seed for reproducibility. Defaults to 7.

        Returns:
            None

        Note:
            The class constructor initializes the trainer with the specified hyperparameters and data handling settings.
            The 'labels_' parameter contains a list of labels for the classes to be used during training.
            The 'positive_number' parameter determines the number of positive instances for each label during training.
            The 'pretrained_model' parameter allows for transfer learning using a pre-trained model.
        """
        self.train_path = train_path
        self.save_dir = save_dir
        self.result_path = result_path
        self.num_samples_list = num_samples_list
        self.model_name_initial = model_name_initial
        self.all_class = all_class
        self.batch_size = batch_size
        self.num_iteration = num_iteration
        self.labels_ = labels_
        self.pretrained_model = pretrained_model
        self.positive_number = positive_number
        self.handler = data_handler(data_path=train_path,
                                    positive_number=positive_number,
                                    labels_=labels_,
                                    load_xlsx=load_xlsx,
                                    SEED=SEED)
        self.logger = logger
        self.pretrained_model_list = pretrained_model_list
    

    def train_by_case(self, label: str = None, verdict_test: list = None, test_path: str = None):
        """
        Train a SetFit model by case.

        This method performs model training for a specific label by separating a given case case from the dataset for validation.
        The training data is loaded from the provided 'test_path', and the specified 'verdict_test' is used as the validation case.
        The model is trained using the remaining data as the training set, and validation is performed on the 'verdict_test' case.
        The training process is logged, and evaluation metrics are computed for the validation case.

        Parameters:
            label (str, optional): The label for which the SetFit model is to be trained. Defaults to None.
            verdict_test (str, optional): The case case to be used as the validation dataset. Defaults to None.
            test_path (str, optional): The file path for the test data. Defaults to None.

        Returns:
            None

        Note:
            The method assumes that the SetFit trainer has been properly initialized with the required parameters for model training.
            The 'test_path' parameter provides the path to the test data, which is used to create a data handler for loading data.
            The training and validation datasets are prepared based on the specified 'verdict_test' case.
            The model is trained using the SetFitTrainer class, and evaluation metrics are computed for the validation case.
        """
        
        handler = data_handler(data_path=self.train_path,
                               positive_number=16,
                               labels_=self.labels_,
                               load_xlsx=True,
                               SEED=7)
        
        # Took from specific label
        df_test = handler.dfs[label]
        df_test['case'] = handler.dfs['case']
        val_df = df_test[df_test['case'].isin(verdict_test)]

        # Train handler
        df = self.handler.dfs[label]
        df['case'] = self.handler.dfs['case']
        train_df = df[~df['case'].isin(verdict_test)]

        self.positive_number = train_df[train_df['label'] == 1].sum()['label']

        train_df = train_df.drop('case', axis=1)
        val_df = val_df.drop('case', axis=1)

        train_ds, val_ds = load_datasets(train_df,
                                         val_df)

        model = SetFitModel.from_pretrained(self.pretrained_model).to(device)
        model_name = str(label) + '-' + str(self.model_name_initial)

        logger_w = create_setfit_logger(model_name,
                                        1,
                                        self.positive_number,
                                        self.batch_size,
                                        self.num_iteration)

        trainer = SetFitTrainer(
            model=model.to(device),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            batch_size=self.batch_size,
            num_iterations=self.num_iteration,
            num_epochs=1
            # loss_class=logger_w
        )

        trainer.train()
        torch.cuda.empty_cache()

        precision, recall, f1, auc_pr, model, val_df = evaluate(model, val_df, str(verdict_test) + '.csv', label)
        return precision, recall, f1, auc_pr, model, val_df


    def train(self, experiment_name):
        today = datetime.date.today().strftime('%Y-%m-%d-%H-%M')
        save_dir_name = today + '_' +experiment_name
        save_path = os.path.join(self.save_dir, save_dir_name)
        # List all files in the directory
        for pretrained_model in self.pretrained_model_list:

            results = {
                "label": [],
                "name": [],
                "precision": [],
                "recall": [],
                "F1-score": [],
                "AUC-PR": []
            }
            df = pd.read_csv(self.train_path)
            tagged_cases = sorted(list(set(df['Case'].values)))
            verdict_sets=[]

            for label in self.labels_:
                self.logger.info(f"ֿ\nStart to train {label} classifier\n")
                start_idx = 0
                while start_idx < len(tagged_cases):
                    verdict_test = tagged_cases[start_idx:start_idx+len(tagged_cases)-1]
                    verdict_sets.append(verdict_test)

                    precision, recall, f1, auc_pr, model, val_df = self.train_by_case(label=label,
                                                                                      verdict_test=verdict_test,
                                                                                      test_path=self.train_path)
                    dfs.append((val_df, str(verdict_test)))
                    results['label'].append(label)
                    results['name'].append(str(verdict_test))
                    results['precision'].append(precision)
                    results['recall'].append(recall)
                    results['F1-score'].append(f1)
                    results['AUC-PR'].append(auc_pr)
                    save_model(save_path, label, str(verdict_test), model, "set-fit", experiment_name)
                    start_idx += 4

            df = pd.DataFrame(results)
            agg_df = df.groupby('label').mean(numeric_only=True).reset_index()

            file_name = f"setfit_{experiment_name}_report.csv"
            result_path = os.path.join(save_path ,file_name)
            agg_df.to_csv(result_path)
            
            self.logger.info(f'The models and their evaluation were successfully saved in {save_path}')
        return save_path