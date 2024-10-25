# COMP90042 Project Codes - Instructions

## Data Processing

The original 5 json data file is expected to be included in a path called "data_raw/". Then we can run all the codes in "data_processor.ipynb". This will produce processed data as csv files in a path called “data_processed/”.

## Task 1

The codes in “task1_method1.ipynb” and “task1_method2.ipynb” will produce the predicted  evidence for claims in the development dataset and the testing set, as csv files in a path called “evdn_pred/”.

The file “task1_method1.ipynb” introduces the method of using sentence matching by TF-IDF vectorization to compare similarities between claims and evidence sentences to find the most relevant evidence for a certain claim.

The file “task1_method2.ipynb” introduces the method of using sentence matching by doc2vec embedding to compare similarities between one claim and another claim. We want to find the most similar labelled claim to a certain test claim, and use the evidence of the most similar labelled claim directly.

## Task 2

The codes in “task2_method1.ipynb” and “task2_method2.ipynb” will predict the label of claims in the development dataset and the testing dataset. We should clarify which evidence prediction in the “evdn_pred/” path is used as evidence prediction for the development dataset and the testing dataset. The codes will directly generate the output json file for the development dataset for evaluation, and for the testing dataset that can be submitted to Codalab. The output files are generated in root path.

The file “task2_method1.ipynb” introduces the claim-only text classification method for fact checking, using BERT plus a classifier.

The file “task2_method2.ipynb” introduces the claim+evidence text classification method for fact checking, using BERT plus a classifier.
