import os
import gzip
import json
import pandas as pd

METADATA_DIR = 'data/meta'
UNITS_LISTS_DIR = METADATA_DIR + '/units-list'
SEQ_DIR = 'data/seq'
UNITS_LIST_FILE_EXT = '.txt'
DATAUNIT_FILE_EXT = '.parquet'
METADATA_FILE_EXT = '.parquet'

GALSON_2015a_STUDY_NAME = 'Galson_2015a'
GALSON_2016_STUDY_NAME = 'Galson_2016'


def load_data(study):
    """
    Loads data from given study.
    
    Args:
        study    name of a study to be loaded.

    Returns:
        dataframe of sequences and metadata from given study and dataframe
        consisting of dataunits metadata from given study.
    """
    
    dataunits_list_path = f'{UNITS_LISTS_DIR}/{study}{UNITS_LIST_FILE_EXT}'

    with open(dataunits_list_path) as dataunits_list_file:
        dataunits = [line.strip() for line in dataunits_list_file.readlines()]

    ddfs = []
    meta_dfs = []
    for dataunit in dataunits:
        data_path = f'{SEQ_DIR}/{study}/{dataunit}{DATAUNIT_FILE_EXT}'
        df = pd.read_parquet(data_path)
        
        metadata_path = f'{METADATA_DIR}/{study}/{dataunit}{METADATA_FILE_EXT}'
        meta_df = pd.read_parquet(metadata_path)
        
        for col in meta_df.columns:
            df[col] = meta_df[col][0]

        ddfs.append(df)
        meta_dfs.append(meta_df)
    
    data = pd.concat(ddfs, axis=0)
    metadata = pd.concat(meta_dfs, axis=0)

    return data, metadata
