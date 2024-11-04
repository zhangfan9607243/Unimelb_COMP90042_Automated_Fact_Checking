# Unimelb COMP90042 Automatic Fact Checking - Climate Change Claims

## Acknowledgement
I would like to extend my sincere gratitude to the Unimelb COMP90042 2023S1 teaching team for providing me with the opportunity to work on this project, as well as for their guidance and feedback on my work.

## Project Introduction

This project focuses on building a automatic fact-checking system for climate science claims.

The goal of this project is to create an automated fact-checking system that can:

1. Task 1: Retrieve relevant evidence from a knowledge base based on the input claim.
2. Task 2: Classify the claim as one of the following:
   - `SUPPORTS`: Evidence supports the claim.
   - `REFUTES`: Evidence refutes the claim.
   - `NOT_ENOUGH_INFO`: Evidence is insufficient to determine the truthfulness.
   - `DISPUTED`: Evidence is inconclusive or contradictory.

In this project, we will adopt an approach that combines an encoder-based LLM (RoBERTa) with a decoder-based LLM (LLaMA).

The final system's performance is evaluated using **Codalab**, where both evidence retrieval and claim classification are measured.

## Files Introduction

Please ensure you set up the following additional directory structure before running the project:

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
    * `t1_result.json`: The prediction of task 1 on testing dataset, after running `/task1/t1_model_testing.py`.
    * `t2_result.json`: The prediction of task 2 on testing dataset, after running `/task2/t2_model_testing.py`.
    * `test-claims-predictions.json`: The final prediction result on testing dataset, after running `/task1/t1_model_testing.py` and then `/task2/t2_model_testing.py` and then `/data_process/Data Process d - Prediction.ipynb`.
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

We did virtually no preprocessing, in `/data_process/Data Process a - Preprocessing.ipynb`, because the LLMs we used, whether encoder-based or decoder-based, can handle complex textual contexts. Therefore, we did not perform cleaning, did not remove stopwords, and did not apply stemming or lemmatization.

We processed the data structure so that it could be directly applied to task 1 and task 2. For task 1, in `/data_process/Data Process b - Task 1.ipynb`, we matched each claim-evidence pair into a row with label 1, and performed negative sampling to match non-corresponding claim-evidence pairs into a row with label 0. For task 2, in `/data_process/Data Process c - Task 2.ipynb`, we concatenated all relevant evidence into a single text passage.

Finally, in `/data_process/Data Process d - Prediction.ipynb`, we processed the final prediction result in the format that **Codalab** competition requires.

## Task 0: Filter Evidences

The original evidence file contains 1.27 million evidences, but not all of them are useful for our tasks.

Firstly, some statements are ambiguous in their references, for example:

```
evidence-204475: He is seeking re-election as the Member of Parliament (MP) for Moray.
evidence-61016: She competed for Brazil at the 2000 Summer Olympics in Sydney, Australia.
```

Secondly, some statemens are not related to climate, or more broadly, science or engineering, for example:

```
evidence-168510: Weird Love -- A man discovers that his girlfriend is a were -- caterpillar.
evidence-472063: Lichtenberger has made five World Series of Poker final tables and has won a WSOP bracelet in 2016.
```

Thirdly, some statements are incomplete or meaningless, for example:

```
evidence-904713: | style = ``text-align : center ;'' | MSK (3 hrs) | | Ivanovo, Russia : GB-1
evidence-85800: Jorge Tavares may refer to:
```

Therefore, we retain evidence statements that are only related to our tasks, in `task0/t0_filter_evidences.py`.

In this case, we use a pretrained LLaMA 3.1 8G model without fine-tuning, to help us filter the relevant statements, using the following prompt:

```
You are an expert in climate change. Please assess whether the following text meets all of the criteria below:
1. The statement is complete and meaningful.
2. The statement includes clear and specific references.
3. The statement is related to climatology, meteorology, geology, or broadly within the fields of physics, chemistry, biology, or engineering.
  
Text: '{evidence}'

Respond with only one word: Yes or No. Please do not respond with anything else.
```

Note: Please ensure that you have permission to use the LLaMA model.

## Task 1: Evidence Retrieval

In this task, we train a distilled RoBERTa model, to learn the relationship between claim and its related evidences, in order to identify whether a evidence is the relevant to a given claim.

* In `t1_dataset_datalod.py`, we define the datasets and data loaders of training and testing data for the model.
* In `t1_model_structure.py`, we define the model structure, which is a distilled RoBERTa model plus a classifier on 2 classes.
* In `t1_model_training.py`, we train the model on training dataset and evaluate the model on validation dataset, with an early stopping mechanism.
* In `t1_model_testing.py`, we use the model to predict on testing dataset, comparing each claim with every filtered evidence.

The predicte result is going to be stored as `/data3/t1_result.json`.

## Task 2: Claim Classification

In this task, we also train a distilled RoBERTa model, to identify the label of a claim, using the combined text of claim and its relevant evidences.

* In `t2_dataset_datalod.py`, we define the datasets and data loaders of training and testing data for the model.
* In `t2_model_structure.py`, we define the model structure, which is a distilled RoBERTa model plus a classifier on 4 classes.
* In `t2_model_training.py`, we train the model on training dataset and evaluate the model on validation dataset, with an early stopping mechanism.
* In `t2_model_testing.py`, we use the model to predict on testing dataset, classifying each claim into one of the four categories.

The predicte result is going to be stored as `/data3/t2_result.json`.

## Final Result

The F1-Score for task 1 (evidence retrieval) and Accuracy score for task 2, as well as the harmonic mean of these two metricses are:

* Task 1: F1-Score (F): 0.3565
* Task 2: Accuracy (A): 0.6837
* Overall: Harmonic Mean of F and A: 0.4686
