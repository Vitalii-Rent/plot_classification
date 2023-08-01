import click
import logging
from pathlib import Path
import pandas as pd
import sys
sys.path.append('src')
from data.text_processing import clean_text, filter_rare, replace_labels, replacements
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv()
SEED=os.getenv('SEED')


def filter_clean_target(input_filepath):
    df = pd.read_csv(input_filepath)
    df['Genre'] = df['Genre'].apply(clean_text)
    df['Genre'] = df['Genre'].apply(lambda x: 'unknown' if x == '' else x)
    df['Genre'] = df['Genre'].apply(lambda x: x.split())
    df = filter_rare(df)
    df['Genre'] = df['Genre'].apply(lambda x: ' '.join(x))
    df = df[df['Genre'] != 'unknown']
    df = df.reset_index()
    df['Genre'] = df['Genre'].apply(lambda x: replace_labels(x, replacements))
    return df
    


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    # Process the data
    processed_data = filter_clean_target(input_filepath)

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
