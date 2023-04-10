# Data Agnostic Topic Modelling (DATM)
A technique for improving topic discovery via data transformation

> A demonstration of the steps necessary for transformation and evaluation is provided in following jupyter notebook: 
> * demonstration.ipynb (in the main folder)

The DATM algorithm, as well as, functions to assist with preprocessing, transformation, and evaluation are included in this project. The folder structure is provided below.



## Data Folder

This folder contains the all the datasets (airline, maccas, 20 news group) - captured in various states, from raw data to transformed data:
* raw data - original dataset
* processed data - preprocessed and cleaned dataset
* transformed data - dataset after applying DATM
* results - contains actual topics

## src Folder
### The preprocessing: 
* **sub folder**: data
* **file**: PreprocessAirlineData
* **function**: preprocessAirline


This function, which is exemplary, is used to preprocess and clean the airline data under the main **Data** folder

### The transformation:
* **sub folder**: models
* **file**: modelling
* **function**: malg

This main DATM function required to perform 
the data transformation. 


### Evaluation
* **sub folder**:evaluations
* **file**: evaluation
* **function**: evaluate_correctness

Used to determine the Purity, Precision, and Recall of topics from models by comparing them 
with the actual topics

## Example
A demonstration of the steps necessary for transformation and evaluation is provided in following jupyter notebook:

* demonstration.ipynb (in the main folder)

## Citation

Bewong, M., Wondoh, J., Kwashie, S., Liu, J., Liu, L., Li, J., Islam, M., & Kernot, D. (2023). DATM: A Novel Data Agnostic Topic Modelling Technique with Improved Effectiveness for both Short and Long Text. IEEE Access, DOI: 10.1109/ACCESS.2023.3262653.

### Paper Link
https://ieeexplore.ieee.org/document/10083137?source=authoralert


