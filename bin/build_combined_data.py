import argparse
import pandas as pd

from data import load_data


def save_combined_data(studies, out_path):
    """
    Constructs and saves combined sequence and metadata dataframe.

    Args:
        studies     names of studies for which combined data should be built.
        out_path    path where combined data dataframe should be saved.
    """

    studies_dfs = [load_data(study)[0] for study in studies]
    concatenated_df = pd.concat(studies_dfs, axis=0)
    concatenated_df.to_parquet(out_path, compression='gzip', engine='pyarrow')


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Constructs and saves combined sequence and metadata dataframes.')
    parser.add_argument('--studies', action='store', nargs='+', help='Names of studies for which combined data should be built.')
    parser.add_argument('--out_data', action='store', help='Path where built dataframe should be saved.')

    args = parser.parse_args()

    save_combined_data(args.studies, args.out_data)