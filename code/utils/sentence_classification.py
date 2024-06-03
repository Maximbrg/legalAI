import os
import datetime
import pandas as pd
import wandb
from torch import nn
from sentence_transformers.losses import CosineSimilarityLoss
from dataclasses import dataclass
from typing import Callable
import functools
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve


def save_model(save_dir, label, model_name, model, train_model_name, experiment_name):
    """

    Save the trained SetFit model and related results to a specified directory. 
    The model is saved using the '_save_pretrained' method of the trainer's model,
    in a 'model' subdirectory within a date-named folder. If a folder with the same date and model name exists, 
    a version number is appended to the folder name.


    Parameters:
        trainer: The SetFitTrainer object containing the trained model.
        model_name (str): The name of the model to be saved.

    """
    
    # Create date folder
    today = datetime.date.today().strftime('%Y-%m-%d')
    version = 1
    save_path = f"{save_dir}/{train_model_name}/{label}/{model_name}"
    while os.path.exists(save_path):
        version += 1
        save_path = f"{save_dir}/{train_model_name}/{label}/{today}-v{version}/{model_name}"
        
        # Save to CSV
        data = {
            'Experiment Name': [experiment_name],
            'Label': [label],
            'Model Name': [train_model_name],
            'Version': [version]
        }

        df = pd.DataFrame(data)
        csv_path = os.path.join(save_dir, 'experiment_names.csv')

        # Check if the CSV exists
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)

            # Filter rows that match the triple
            mask = (df_existing['Experiment Name'] == experiment_name) & (df_existing['Label'] == label) & (df_existing['Model Name'] == train_model_name) & (df_existing['Version'] == version)

        # If the row does not exist in the CSV
            if not mask.any():
                df = pd.concat([df_existing, df], ignore_index=True)
                df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)
    else:
        os.makedirs(save_path)

    # Save model
    model.save_pretrained(save_path)



def create_setfit_logger(model_name, num_epochs, num_samples, batch_size, num_iterations):
    """Create a Hugging Face Trainer logger that logs training loss and metrics to Weights & Biases."""

    @dataclass
    class LoggingWrapper:
        loss_class: nn.Module
        num_epochs: int
        num_samples: int
        batch_size: int
        num_iterations: int

        def __call__(self, *args, **kwargs):
            wandb.init(project="setfit", name=model_name)
            wandb.config.update({"num_epochs": self.num_epochs, "num_samples": self.num_samples,
                                 "batch_size": self.batch_size, "num_iterations": self.num_iterations},
                                allow_val_change=True)
            loss_class_instance = self.loss_class(*args, **kwargs)
            loss_class_instance.forward = self.log_forward(loss_class_instance.forward)
            # loss_class_instance.validate = self.log_validation_loss(loss_class_instance.validate)
            return loss_class_instance

        def log_forward(self, forward_func: Callable):
            @functools.wraps(forward_func)
            def log_wrapper_forward(*args, **kwargs):
                loss = forward_func(*args, **kwargs)
                wandb.log({"training_loss": loss, "num_epochs": self.num_epochs, "num_samples": self.num_samples,
                           "batch_size": self.batch_size, "num_iterations": self.num_iterations})
                return loss

            return log_wrapper_forward

        def log_validation_loss(self, validate_func: Callable):
            @functools.wraps(validate_func)
            def log_wrapper_validate(*args, **kwargs):
                loss = validate_func(*args, **kwargs)
                wandb.log(
                    {
                        "validation_loss": loss,
                        "num_epochs": self.num_epochs,
                        "num_samples": self.num_samples,
                        "batch_size": self.batch_size,
                        "num_iterations": self.num_iterations,
                    }
                )
                return loss

            return log_wrapper_validate

    return LoggingWrapper(CosineSimilarityLoss, num_epochs, num_samples, batch_size, num_iterations)


def load_datasets(train_df, val_df):
    """Load the training and validation datasets from the Pandas DataFrames."""
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return train_ds, val_ds

def evaluate(model, tagged_df, case_id=None, label=None, logger=None):

        """
        Evaluate the SetFit model on the validation dataset and save the results.

        This method evaluates the SetFit model on the provided validation dataset and computes various evaluation metrics.
        The evaluation results are saved in a CSV file with the specified 'name' in the given directory.
        The evaluation metrics include precision, recall, F1-score, and AUC-PR (Area Under the Precision-Recall curve).

        Parameters:
            model: The trained SetFit model to be evaluated.
            tagged_df (pd.DataFrame): The validation DataFrame containing the validation dataset.
            name (str): The name of the CSV file to save the evaluation results.

        Returns:
            None

        Note:
            The method assumes that the SetFit model has been trained and is ready for evaluation.
            The 'val_df' parameter is a DataFrame containing the validation dataset with columns 'text' and 'label'.
            The evaluation metrics are computed based on the model's predictions on the validation data and the ground truth labels.
            The evaluation results are saved in a CSV file with the specified 'name' in the given directory.
        """

        try:
            y_true = tagged_df['label'].values
        except:
            y_true = tagged_df[label].values
            
        y_pred = model.predict_proba(tagged_df['text'].values).numpy()[:, 1]
        tagged_df['proba'] = y_pred
        tagged_df['case_id'] = case_id
        tagged_df['label_predict'] = label

        y_pred_round = model.predict_proba(tagged_df['text'].values).numpy().argmax(axis=1)
        precision = precision_score(y_true
                                    ,y_pred_round)
        recall = recall_score(y_true, y_pred_round)
        f1 = f1_score(y_true, y_pred_round)
        precision_1, recall_1, thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = auc(recall_1, precision_1)
        if logger is not None:
            logger.info(f"for label {label}: \n Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-PR: {auc_pr}")
        else:
            print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-PR: {auc_pr}")
        if label == 'reject':
            tagged_df = tagged_df.loc[tagged_df.index[y_pred_round] != 1]
        return precision, recall, f1, auc_pr, model, tagged_df