from datetime import datetime
import os
import sys
import pandas as pd

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from predict_sentence import predict_2cls_lvl_flow
from utils.files import config_parser, setup_logger
from utils.sentence_classification import evaluate


def evaluation(params, logger, dir_path):
    """
    This function goes through the tagging file, a legal contract and performs an evaluation
    """ 
    # TODO: generalized to config
    # TODO: run on list of files/dir and mean the results 
    tagged_path = params['tagged_path']
    eval_path = params['eval_path']

        # Initialize accumulators for metrics
    metric_sums = {}
    file_counts = {}
    sentence_counts = {}
    case_count = []
    _, _, classifiers = predict_2cls_lvl_flow(tagged_path=tagged_path,
                                              eval_path=eval_path,
                                              classifiers_path=params['classifiers_path'],
                                              result_path=params['result_path'],
                                              logger=logger,
                                                threshold=params['threshold']
                                              )
    today = datetime.today()
    formatted_date = today.strftime("%d.%m")
    save_path = os.path.join(eval_path, formatted_date)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in os.listdir(dir_path):
        if 'ME' in file:

            tagged_df = pd.read_csv(os.path.join(dir_path, file))
            save_path = os.path.join(save_path, f"{file}.txt")

            all_metrics = ""
            for column in tagged_df.columns:
                if column == 'CIRCUM_OFFENSE':
                    continue
                try:
                    column_upper = column.upper()
                    if column == 'CONFESSION':
                        column_upper = 'CONFESSION_LVL2'
                    if column == 'verdict':
                        case_count.extend(list(tagged_df[column].values))
                    amount_value = sum(tagged_df[column].values)
                    sentence_count = tagged_df[column].sum()

                    precision, recall, f1, auc_pr, model, tagged_df = evaluate(model=classifiers[column_upper],
                                                                               tagged_df=tagged_df,
                                                                               label=column, 
                                                                               logger=logger)
                    
                    # Initialize metrics accumulators for this column if not already initialized
                    if column_upper not in metric_sums:
                        metric_sums[column_upper] = {'precision': 0, 'recall': 0, 'f1': 0, 'auc_pr': 0}
                        file_counts[column_upper] = 0
                        sentence_counts[column_upper] = 0


                    # Accumulate metrics
                    metric_sums[column_upper]['precision'] += precision
                    metric_sums[column_upper]['recall'] += recall
                    metric_sums[column_upper]['f1'] += f1
                    metric_sums[column_upper]['auc_pr'] += auc_pr
                    file_counts[column_upper] += 1
                    sentence_counts[column_upper] += sentence_count

                    metrics_text = (f"Label: {column}\n"
                                    f"Precision: {round(precision, 3)}\n"
                                    f"Recall: {round(recall, 3)}\n"
                                    f"F1 Score: {round(f1, 3)}\n"
                                    f"AUC-PR: {round(auc_pr, 3)}\n"
                                    f"Amount true labels: {amount_value}\n"
                                    f"Sentence count: {sentence_count}\n"  # Add sentence count information
                                    "-----------------------\n")
                    all_metrics += metrics_text
                    
                except Exception as e:
                    logger.error(f"Error evaluating column {column}: {e}")
                    continue

            # with open(save_path, 'w') as file:
            #     file.write(all_metrics)

            logger.info(f"All metrics saved to {save_path}")

    # Compute average metrics per column
    avg_metrics_text = "Average Metrics per Column:\n"
    for column_upper, metrics in metric_sums.items():
        if file_counts[column_upper] > 0:
            avg_metrics = {key: value / file_counts[column_upper] for key, value in metrics.items()}
            avg_metrics_text += (f"Label: {column_upper}\n"
                                 f"Precision: {round(avg_metrics['precision'], 3)}\n"
                                 f"Recall: {round(avg_metrics['recall'], 3)}\n"
                                 f"F1 Score: {round(avg_metrics['f1'], 3)}\n"
                                 f"AUC-PR: {round(avg_metrics['auc_pr'], 3)}\n"
                                 f"Cases count: {len(set(case_count))}\n"
                                 f"Sentence count: {sentence_counts[column_upper]}\n"  
                                 "-----------------------\n")

    # Save average metrics
    avg_save_path = os.path.join(eval_path, formatted_date, "average_metrics.txt")
    with open(avg_save_path, 'w') as avg_file:
        avg_file.write(avg_metrics_text)
    
    logger.info(f"Average metrics saved to {avg_save_path}")

def main(params):
    logger = setup_logger(save_path=os.path.join(params['result_path'], 'logs'),
                          file_name='predict_sentence_cls_test')
    evaluation(params,
               logger,
               dir_path = params['dir_path'])



if __name__ == '__main__':
    param = config_parser('/home/maximbr/legalAI/legalAI/code/configs/main_config')
    main(param)