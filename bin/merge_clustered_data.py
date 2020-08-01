import os
import argparse
import pandas as pd
from bin.build_clustered_data import CLUSTER_ID_COLUMN_NAME

CLUSTERED_FILE_EXT = '.parquet'


def save_merged_clustered_data(df_dir, out_path):
    """
    Merges partial clustered data and saves it.
    
    Args:
        df_dir    paths to a directory containing the clustered group data to be merged.
        out_path  path where merge clustered data should be saved.
    """

    cnt = 0
    partial_clustered_dfs = []

    print(f'Loading data in "{df_dir}"...')
    files_in_dir = sorted(os.listdir(df_dir))
    for filename in files_in_dir:
        if filename.endswith(CLUSTERED_FILE_EXT):
            df = pd.read_parquet(f'{df_dir}/{filename}')
            
            # Adjust cluster IDs in order to get unique cluster IDs
            df[CLUSTER_ID_COLUMN_NAME] = df[CLUSTER_ID_COLUMN_NAME] + cnt
            cnt = cnt + df[CLUSTER_ID_COLUMN_NAME].nunique()

            partial_clustered_dfs.append(df)

    print('Merging data...')
    merged_cluster_data = pd.concat(partial_clustered_dfs, axis=0)

    print('Saving output data...')
    merged_cluster_data.to_parquet(out_path)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Merges partial (per group) clustered data and saves it.')
    parser.add_argument('--data_dir', action='store', help='Paths to a directory which contains the clustered group data to be merged.')
    parser.add_argument('--out_data', action='store', help='Path where merged clustered dataframe should be saved.')

    args = parser.parse_args()

    save_merged_clustered_data(args.data_dir, args.out_data)
