import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import sys
sys.path.append('src')
from data.text_processing import clean_text, lemming, remove_stopwords
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords



def process_data(input_filepath):
    df = pd.read_csv(input_filepath)
    en_stopwords = stopwords.words('english')
    df['Plot'] = df['Plot'].apply(clean_text)
    df['Plot'] = df['Plot'].apply(lambda X: word_tokenize(X))
    df['Plot']=df['Plot'].apply(lambda x: remove_stopwords(x, en_stopwords))
    df['Plot']=df['Plot'].apply(lemming)
    return df
    


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn interim data from (../interim) into
        cleaned and processed data ready to be analyzed and used in models (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('processing texts before creating embeddings')


    # Process the data
    processed_data = process_data(input_filepath)

    # Save the processed data to the output file
    processed_data.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
