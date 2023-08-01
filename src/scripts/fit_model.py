import click
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from data.text_processing import  dummy, partial_clean_text, replacements, best_classes, replace_labels
from models.eval_nn_utils import compute_metrics
from models.train_predict import train_evaluate_mll
from data.nn_utils import MovieHFDatasetMLL
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
import os
load_dotenv()
SEED=int(os.getenv('SEED'))

def fit_log_reg():
    df = pd.read_csv('data\processed\data_processed_shortened.csv')
    mlb = MultiLabelBinarizer()
    shortened_classes = mlb.fit_transform(df['Genre'])
    vectorizer = joblib.load(r'models/utils/vectorizer.pkl')
    X_train, X_test, y_train, y_test = train_test_split(df.Plot,
                                                shortened_classes, shuffle=True, random_state=SEED, stratify=shortened_classes)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    model = ClassifierChain(LogisticRegression(n_jobs=-1, random_state=SEED, C=4.3))
    model = train_evaluate_mll(model, X_train, X_test, y_train, y_test)
    return model


def llm_process(df):
    labels = df['Genre'].apply(lambda x: replace_labels(x, replacements))
    plots_cleaned = df['Plots'].apply(partial_clean_text)
    genres_shortened = labels.apply(lambda labels: list(filter(lambda label: label in best_classes , labels)))
    data_shortened = pd.concat([plots_cleaned, genres_shortened], axis=1)
    data_shortened = data_shortened[data_shortened['Genre'].apply(lambda x: len(x) > 0)]
    return data_shortened


def fit_bert():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('../data/interim/data_intermidiate.csv')
    df = llm_process(df)

    X_train, X_val, y_train, y_val = train_test_split(df.Plot, df.Genre, test_size=0.2, stratify=df.Genre)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    X_train_tokenized = tokenizer(X_train.to_list(), truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val.to_list(), truncation=True, max_length=512)

    mlb = MultiLabelBinarizer()
    one_hot_train = mlb.transform(y_train)
    one_hot_val = mlb.transform(y_val)
    id2label = {idx:label for idx, label in enumerate(mlb.classes_)}
    label2id = {label:idx for idx, label in enumerate(mlb.classes_)}

    train_dataset = MovieHFDatasetMLL(X_train_tokenized, one_hot_train, label2id)
    val_dataset = MovieHFDatasetMLL(X_val_tokenized, one_hot_val, label2id)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(mlb.classes_))
    training_args = TrainingArguments(
        output_dir="Bert_clf",
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer
    


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('model_type', type=click.Choice(['log_reg', 'bert'], case_sensitive=False))
def main(output_filepath, model_type):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('predict labels of movie plots')
    if model_type == 'log_reg':
        model = fit_log_reg()
        joblib.dump(model, output_filepath)
    elif model_type == 'bert':
        trainer = fit_bert()
        trainer.save_model(output_filepath)
    else:
        print('Incorrect model type, choose between log_reg and bert')
        exit()

    # Save the model to the specified file
    
    print('Model saved successfully')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
