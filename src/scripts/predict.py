import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from data.text_processing import  process_script, dummy, partial_clean_text
import joblib
import torch
from transformers import AutoTokenizer, DistilBertConfig, AutoModelForSequenceClassification
import os
load_dotenv()
SEED=os.getenv('SEED')


def predict_log_reg(input_filepath):
    if input_filepath.lower().endswith('.txt'):
        with open(input_filepath, 'r') as file:
            text = file.read()
            df = pd.Series(text)
    elif input_filepath.lower().endswith('.csv'):
        df = pd.read_csv(input_filepath)
    else:
        print("Invalid file type. Please provide a .txt or .csv file.")
    df = process_script(df)
    
    vectorizer = joblib.load(r'models/utils/vectorizer.pkl')
    df = vectorizer.transform(df)
    model = joblib.load(r'models\multilabel_models\ML models\log_reg_mll_reduced')
    preds = model.predict(df)
    return preds


def predict_bert(input_filepath):
    if input_filepath.lower().endswith('.txt'):
        with open(input_filepath, 'r') as file:
            text = file.read()
            df = pd.Series(text)
    elif input_filepath.lower().endswith('.csv'):
        df = pd.read_csv(input_filepath)
    else:
        print("Invalid file type. Please provide a .txt or .csv file.")
    
    df = df.apply(partial_clean_text)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    folder_path = r"models\multilabel_models\NN models\bert_multi_label_shortened"
    config = DistilBertConfig.from_pretrained(folder_path)
    model = AutoModelForSequenceClassification.from_pretrained(folder_path, config=config)

    inputs = tokenizer(df.to_list(), truncation=True, max_length=512, return_tensors="pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_probabilities = torch.sigmoid(outputs.logits)
    # Get the predicted class label
    threshold = 0.5
    predicted_classes = (predicted_probabilities > threshold).int().detach().cpu().numpy()
    return predicted_classes


def decode_preds(preds):
    mlb = joblib.load(r'models/utils/binarizer.pkl')
    classes_list = []

    for sample in preds:
        indexes_of_ones = np.where(sample == 1.)[0]
        classes = [mlb.classes_[idx] for idx in indexes_of_ones]
        classes_list.append(np.array(classes))
    
    return np.array(classes_list)
    


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_type', type=click.Choice(['log_reg', 'bert'], case_sensitive=False))
def main(input_filepath, output_filepath, model_type):

    logger = logging.getLogger(__name__)
    logger.info('predict labels of movie plots')
    if model_type == 'log_reg':
        predictions = predict_log_reg(input_filepath)
    elif model_type == 'bert':
        predictions = predict_bert(input_filepath)
    else:
        print('Incorrect model type, choose between log_reg and bert')
        exit()
    predicts_words = decode_preds(predictions)


    # Save the processed data to the output file
    np.savetxt(output_filepath + '/deceded.txt', predicts_words, delimiter=', ', fmt='%s')
    np.save(output_filepath + '/predictions', predictions)
    print('Predictions saved successfully')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
