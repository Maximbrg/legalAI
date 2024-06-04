
# Criminal Case Classification Project

This repository houses a project focused on classifying key aspects of criminal cases within the Israeli legal framework. The project leverages a few-shot learning approach for accurate sentence classification relevant to sentencing decisions.

### Key Features

* **Code:** Implements a few-shot learning approach for accurate sentence classification.
* **Data:** Created in collaboration with Israeli Ministry of Justice criminal law experts, covering crucial aspects of criminal cases.
* **Evaluation:** Includes an evaluation of the applied methodology for classifying sentences in weapon-related cases.

### Execution

The project offers two primary functionalities through scripts located within the `code/flows` directory:

1. **Train Classifiers:** Train new classifiers using the script `code/flows/train_sentence.py`. The default configuration file is provided at `code/configs/train_sentence_cls.yaml`.
2. **Evaluate Classifiers:** Evaluate previously trained classifiers using the script `code/flows/predict_sentence.py`. The default configuration file is located at `code/configs/train_sentence_cls.yaml`.

### Data

The `data/sentence_classification` folder contains the following files:

* **ME_sentence_tag_train.csv:** This file holds sentences tagged by two criminal law experts. Any disagreements between the experts were resolved through discussion.
* **ME_sentence_tag_test.csv:** This file contains sentences tagged by two data science researchers receiving instructions directly from legal experts. Disagreements between researchers were resolved through joint discussion.
* **statistics.md:** This file provides statistics regarding the datasets used within the project.
* **scheme.md:** This file outlines the classification scheme collaboratively developed by criminal law experts and data science researchers.

# Results
The results are sorted in: `results/evaluations/sentence_calssification/04.06/average_metrics.txt`
