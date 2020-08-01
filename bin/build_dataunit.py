import os
import pathlib
import gzip
import json
import argparse
import pandas as pd

def parse_data(dataunit_path):
    """
    Parses data from given dataunit.
    
    Args:
        dataunit_path    path to a dataunit to be parsed.
        
    Returns:
        parsed data and metadata of a dataunit.
    """

    metadata_line = True
    data = []
    for line in gzip.open(dataunit_path, 'rb'):
        line_data = json.loads(line)

        if metadata_line:
            metadata = line_data
            metadata_line = False
        else:
            data.append(line_data)

    return data, metadata

def create_intermediate_dirs(path):
    """
    Creates intermediate directories of a path if they do not exist.

    Args:
        path    path whose intermediate directories should be created.
    """

    path_dir = pathlib.Path(os.path.dirname(path))
    path_dir.mkdir(parents=True, exist_ok=True)

def save_as_parquets(dataunit_path, out_data, out_metadata):
    """
    Saves data and metadata from given dataunit as parquets.
    
    Args:
        dataunit_path    path to a dataunit whose data and metadata should be saved.
        out_data         path where to save data.
        out_metadata     path where to save data.
    """
    
    data, metadata = parse_data(dataunit_path)

    create_intermediate_dirs(out_data)
    create_intermediate_dirs(out_metadata)
    
    pd.DataFrame(data).to_parquet(out_data, compression='gzip', engine='fastparquet')
    pd.DataFrame([metadata]).to_parquet(out_metadata, compression='gzip', engine='fastparquet')


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Build parquet file for given dataunit.')
    parser.add_argument('--dataunit', action='store', help='Path of a dataunit for which parquet files should be built.')
    parser.add_argument('--out_data', action='store', help='Path where output data should be saved.')
    parser.add_argument('--out_metadata', action='store', help='Path where output metadata should be saved.')

    args = parser.parse_args()

    save_as_parquets(args.dataunit, args.out_data, args.out_metadata)
