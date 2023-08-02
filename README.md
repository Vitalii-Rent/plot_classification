
# `Plot classification`

## Objective

The objective is classification of genres based on plot section from wikipedia. More broadly this project allows any sort of classification of texts for multiclass or multilabel problems with little or no changes in code for a wide range of models - classical ml, recurent nn or transformers.

## Explorative results

Results are following - multiclasss models achieved highest full match scores while multilabel - jaccard - intersection with target. I've tried a number of models including nn ones - LSTM and BERT. The best models based on metrics in both mlc and mll appeared to be fine-tuned berts the next model is single or chain of logistic regressions. Only those two models were also fit on a specific subset of labels containing only the most frequent ones. Both of these are included in predict scripts considering log_reg model is significantly faster and easier to fit. 

## Modeling results
Columns description :
- accuracy in this context is a number of correctly predicted samples/all samples, in multi-label case full-match accuracy is meant
- jaccard score - in multiclass context is the same as accuracy, in multilabel context - number of individually correctly classified labels devided by all samples - an adequate equivalent of accuracy for multilabel models.
- f1_weighted - weighted sum of f1 scores computed for each class.

Results were gained on unseen data with a wide range of labels.


|                                             |   accuracy |   f1_weighted |   jaccard |
|:--------------------------------------------|-----------:|--------------:|----------:|
| ('multi_label', 'bert_tuned')               |      0.396 |         0.508 |     0.471 |
| ('multi_class', 'bert_tuned')               |      0.462 |         0.4   |     0.462 |
| ('multi_label', 'log_reg_chain_tuned')      |      0.36  |         0.43  |     0.427 |
| ('multi_class', 'log_reg_tuned')            |      0.41  |         0.325 |     0.41  |
| ('multi_class', 'log_reg')                  |      0.391 |         0.302 |     0.391 |
| ('multi_label', 'lgbm_chain')               |      0.292 |         0.43  |     0.381 |
| ('multi_label', 'log_reg_chain')            |      0.313 |         0.36  |     0.368 |
| ('multi_class', 'lstm_pretrained_word2vec') |      0.368 |       nan     |     0.368 |
| ('multi_class', 'lstm_fasttext')            |      0.367 |       nan     |     0.367 |
| ('multi_label', 'ridge_tuned')              |      0.242 |       nan     |     0.342 |
| ('multi_class', 'random_forest')            |      0.327 |         0.234 |     0.327 |
| ('multi_class', 'random_forest_tuned')      |      0.327 |         0.232 |     0.327 |
| ('multi_label', 'ridge')                    |      0.261 |         0.39  |     0.313 |
| ('multi_class', 'bayes')                    |      0.23  |         0.086 |     0.23  |
| ('multi_label', 'random_forest')            |      0.069 |       nan     |     0.079 |
| ('multi_label', 'bayes_chain')              |      0.001 |       nan     |     0.001 |

Results were gained on unseen data with a smaller range of the more frequent labels

|         |   accuracy |   f1_score |   jaccard_score |
|:--------|-----------:|-----------:|----------------:|
| log_reg |      0.431 |      0.503 |           0.496 |
| bert    |      0.453 |      0.601 |           0.521 |

## Usage

The main part of project consists of 5 notebooks and 4 scripts. Main usage scenario is dealing with the scripts.

clean_filter_dataset.py is a script for preprocessing data, i.e. converting raw data to intermediate data. It involves cleaning and standardizing labels and filtering out extremely rare and unknown instances.
Usage: `python clean_filter_dataset.py <input file_path> <output file_path>`
Arguments:
  [input file_path]: path to the raw datset that is to be processed
  [output file_path]: path to the place you want to save processed data

process_dataset.py is a script for cleaning and processing of target. It involves cleaning, removing of stop words and lemmatizaing the texts.
Usage: `python process_dataset.py <input file_path> <output file_path>`
Arguments:
  [input file_path]: path to the intermid datset that is to be processed
  [output file_path]: path to the place you want to save processed data

fit_model.py is a script for fitting a model. Currently there are two types of models involved - BERT and chain of logistic regressions
Usage: `python fit_model.py <input file_path> <output file_path>`
Arguments:
  [output file_path]: path to the place you want to save fitted model
  [model type]: type of the model to train, there are 2 options - 'log_reg' and 'bert'

predict.py is a script for using models for predicting labels of plots. Input can be txt file with one instance or csv file with a number of instances. Texts should be completely unprocessed, all the necessary steps are performed inside the script. Outputs numpy array with binarized predictions and txt file with decoded predicted labels. Currently there are two models involved - fine-tuned BERT and chain of logistic regressions.
Usage: `python predict.py <input file_path> <output file_path>`
Arguments:
  [input file_path]: path to data with texts you want to predict
  [output file_path]: path to the place you want to save predictions 
  [model type]: type of the model to train, there are 2 options - 'log_reg' and 'bert'

Example:
`python src/scripts/predict.py data/plot.txt data/result log_reg`
 - script, that was run from the root of the project

## Deploy
- In order to deploy the project you can just install the necessary requirements that are listed in the requirements.txt file:
pip install -r requirements.txt
- or by using docker:
project already has dockerfile and so you have to build the image and create a container in which you can scripts and notebooks.
1. `docker build -t [plot_project] .` - this command builds an image with a name of [plot_project]
2. `docker run -d -it --name [name]  -p 8888:8888  -v $pwd/data:/app/data -v $pwd/models:/app/models [plot_project]` - this command creates a container with name [name], -v $pwd/data:/app/data part mounts data from project directory to the container, $pwd/models:/app/models part mounts models from the project directory to the container. These can be changed if you don't plan to use following folder or if you have specific folder with data or models.

After setup of container is finished, you can run notebooks or scripts the following way:
to run scripts - 

`docker exec -it <name>  python src/scripts/[script].py <first_arg> <second_arg>`
example:
`docker exec -it <name>  python src/scripts/make_dataset.py data/raw/wiki_movie_plots_deduped.csv data/processed/final_test.csv`

to run notebook:

`docker exec -it <name>  jupyter notebook notebooks/<notebook_name>.ipynb  --allow-root --ip=0.0.0.0 --port=8888 `
after running this command, choose one of 2 links - preferably the 'http://127.0.0.1:8888' one. 

example:
`docker exec -it test8  jupyter notebook notebooks/1.0-data-exploration-processing.ipynb  --allow-root --ip=0.0.0.0 --port=8888 `



## Project Organization
------------
A short description of the project.

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been cleaned but have not been transformed
    │   ├── processed      <- The final, canonical data sets for modeling that is cleaned and processed
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, vectorization utils
    │   ├── multiclass_models       <- Models that solve classification problem in multiclasss maner
    │   |   ├── ML      <- Classical machine learning models, i.e. logistic regression, etc.
    │   |   └── NN      <- Neural networks - LSTM and BERT
    │   ├── multilabel_models       <- Models that solve classification problem in multilabel maner
    │   |   ├── ML      <- Simmilar to multiclass
    │   |   └── NN      <- Simmilar to multiclass
    │   └── utils      <- TF-IDF vectorizer
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                     and a short `-` delimited description, e.g.
    │   |                     `1.0-data-exploration-processing`.
    │   ├── 1.0-data-exploration-processing        <- Cleaning and processing of data, i.e. removing stop words, lemmatizaing, etc
    │   ├── 2.0-multi-class-ml                     <- Training and evaluating classical ml models in milti-class manner
    │   ├── 3.0-multi-label-ml                     <- Training and evaluating classical ml models in multi-label manner
    │   ├── 4.0_multi_class_LSTM                   <- Training and evaluating LSTM in multi-class manner
    │   └── 5.0-LLM                                <- Fine-tuning and evaluating BERT model
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- PLots, in specific case - importances of words by class
    │   ├── tables         <- Tables with results
    │   └── tensorboards   <- Tensorboards runs achieved when training LSTM models
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Utils to process data
    │   │   ├── nn_utils.py
    │   │   └── text_processing.py
    |   |    
    │   │
    │   ├── features       <- utils to vectorize text for ml nodels
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Utils to train, evaluate and tune models.                
    │   │   ├── eval_nn_utils.py    
    │   │   ├── train_nn_utils.py
    │   │   ├── train_predict.py
    │   │   └── tune_utils.py
    │   |
    │   ├── visualization  <- Utils to create visualizetions, in precific case - word importances and word cloud space
    │   |   └── visualize.py
    │   └── scripts  <- Scripts to clean and process data, to train and use models.
    │   │   ├── fit_model.py    
    │   │   ├── clean_filter_dataset.py
    │   │   ├── predict.py
    │   │   └── process_dataset.py
--------
> developer `Rentiuk Vitalii` (`vital.rentyk03@gmail.com`)
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
