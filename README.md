# Unimelb COMP90042 Automatic Fact Checking - Climate Change Claims

## Project Introduction

This project focuses on building a automatic fact-checking system for climate science claims.

The goal of this project is to create an automated fact-checking system that can:

1. Task 1: Retrieve relevant evidence from a knowledge base based on the input claim.
2. Task 2: Classify the claim as one of the following:
   - `SUPPORTS`: Evidence supports the claim.
   - `REFUTES`: Evidence refutes the claim.
   - `NOT_ENOUGH_INFO`: Evidence is insufficient to determine the truthfulness.
   - `DISPUTED`: Evidence is inconclusive or contradictory.

The final system's performance is evaluated using **Codalab**, where both evidence retrieval and claim classification are measured.

Please ensure you set up the following directory structure before running the project:

```
/
|---data/
    |---data1/
    |---data2/
    |---data3/
|---model/
    |---model_task1/
    |---model_task2/

```

The following is detailed explanation of folders and files:

* `data/`: The folder for original data, processed data, and the output data

  * `data1/`: The folder for original data, which contains:
    * `train-claims.json`: The training dataset containing the claims, related evidence, and the label.
    * `dev-claims.json`: The validation dataset containing the claims, related evidence, and the label.
    * `dev-claims-baseline.json`: The same validation dataset with predicted label for demostration of evaluation.
    * `test-claims-unlabelled.json`: The testing dataset with claim only.
    * `evidence.json`: The list of evidences.
    * `eval.py`: The function for evaluation.
  * `data2/`: The folder for processed data, which is expected to contain processed training, validation, testing, and evidence data, which is ready for the following tasks. This folder is going to contain:
    * `data_evdn.json`: The simply preprocessed evidences, after running `/data_process/Data Process a - Preprocess.ipynb`.
    * `data_evdn_filtered.json`: The filtered evidences by the help of decoder-based model LLaMA 3.1, after running `/task0/to_filter_evidences.py`.
    * `data_tran.json`: The simply preprocessed training dataset, after running `/data_process/Data Process a - Preprocess.ipynb`.
    * `data_vald.json`: The simply preprocessed validation dataset, after running `/data_process/Data Process a - Preprocess.ipynb`.
    * `data_test.json`: The simply preprocessed testing dataset, after running `Data /data_process/Data Process a - Preprocess.ipynb`.
    * `data_tran_t1.json`: The training dataset that is ready for task 1, after running `/data_process/Data Process b - Task 1.ipynb`.
    * `data_vald_t1.json`: The validation dataset that is ready for task 1, after running `/data_process/Data Process b - Task 1.ipynb`.
    * `data_test_t1.json`: The testing dataset that is ready for task 1, after running `/data_process/Data Process b - Task 1.ipynb`.
    * `data_tran_t2.json`: The training dataset that is ready for task 2, after running `/data_process/Data Process c - Task 2.ipynb`.
    * `data_vald_t2.json`: The validation dataset that is ready for task 2, after running `/data_process/Data Process c - Task 2.ipynb`.
    * `data_test_t2.json`: The testing dataset that is ready for task 2, after running `/task1/t1_model_testing.py` and then `/data_process/Data Process c - Task 2.ipynb`.
  * `data3/`: The folder for final output, which is expected to contain labelled test data for final evaluation, which is going to contain:
    * t1_result.json: The prediction of task 1 on testing dataset, after running `/task1/t1_model_testing.py`.
    * t2_result.json: The prediction of task 2 on testing dataset, after running `/task2/t2_model_testing.py`.
    * test-claims-predictions.json: The final prediction result on testing dataset, after running `/task1/t1_model_testing.py` and then `/task2/t2_model_testing.py` and then `/data_process/Data Process d - Prediction.ipynb`.
* `model/`: The saved language models.

  * `model_task1/model_task1.pth`: The Distilled RoBERTa Model for the first task, after running `/task1/t1_model_training.py`.
  * `model_task2/model_task2.pth`: The Distilled RoBERTa Model for the second task, after running `/task2/t2_model_training.py`.
* `data_process/`: The notebooks for data processing.

  * `Data Process a - Preprocess.ipynb`: Simple data preprocessing on training, validation, testing, and evidence data.
  * `Data Process b - Task 1.ipynb`: Process training, validation, and testing data for task 1, including negative sampling of evidences for training and validation data.
  * `Data Process c - Task 2.ipynb`: Process training, validation, and testing data for task 2.
  * `Data Process d - Prediction.ipynb`: Generate the final prediction result on testing data for project evaluation.
* `task0/t0_filter_evidences.py`: Use pretrained LLaMA 3.1 8G model without fine-tunning to help filter evidences that is only related to the climate topic.
* `task1/`: The codes of task 1.

  * `t1_dataset_datalod.py`: Define dataset and data loader for task 1.
  * `t1_model_structure.py`: Define model for task 1, which is a distilled RoBERTa model plus a classifier.
  * `t1_model_training.py`: Train and save the model for task 1.
  * `t1_model_testing.py`: Use model to predict on testing data for task 1.
* `task2/`: The codes of task 2.

  * `t1_dataset_datalod.py`: Define dataset and data loader for task 1.
  * `t1_model_structure.py`: Define model for task 1, which is a distilled RoBERTa model plus a classifier.
  * `t1_model_training.py`: Train and save the model for task 1.
  * `t1_model_testing.py`: Use model to predict on testing data for task 1.

## Data Processing

The original 5 json data file is expected to be included in a path called "data_raw/". Then we can run all the codes in "data_processor.ipynb". This will produce processed data as csv files in a path called “data_processed/”.

## Task 0

## Task 1

The codes in “task1_method1.ipynb” and “task1_method2.ipynb” will produce the predicted  evidence for claims in the development dataset and the testing set, as csv files in a path called “evdn_pred/”.

The file “task1_method1.ipynb” introduces the method of using sentence matching by TF-IDF vectorization to compare similarities between claims and evidence sentences to find the most relevant evidence for a certain claim.

The file “task1_method2.ipynb” introduces the method of using sentence matching by doc2vec embedding to compare similarities between one claim and another claim. We want to find the most similar labelled claim to a certain test claim, and use the evidence of the most similar labelled claim directly.

## Task 2

The codes in “task2_method1.ipynb” and “task2_method2.ipynb” will predict the label of claims in the development dataset and the testing dataset. We should clarify which evidence prediction in the “evdn_pred/” path is used as evidence prediction for the development dataset and the testing dataset. The codes will directly generate the output json file for the development dataset for evaluation, and for the testing dataset that can be submitted to Codalab. The output files are generated in root path.

The file “task2_method1.ipynb” introduces the claim-only text classification method for fact checking, using BERT plus a classifier.

The file “task2_method2.ipynb” introduces the claim+evidence text classification method for fact checking, using BERT plus a classifier.

## Final Result
