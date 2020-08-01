import os
import argparse
import pandas as pd


RANDOM_STATE = 42


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Subsample single row per group data and saves it.')
    parser.add_argument('--input', action='store', help='Paths to a dataframe to be subsampled.')
    parser.add_argument('--groupby_col', action='store', help='Name of a column to be used for groupby.')
    parser.add_argument('--output', action='store', help='Path where subsampled dataframe should be saved.')

    args = parser.parse_args()

    print('Loading input dataframe...')
    df = pd.read_parquet(args.input)

    print('Subsampling...')
    # Get randomly sampled row per cluster
    df_subs = df.groupby(args.groupby_col).apply(lambda gr: gr.sample(1, random_state=RANDOM_STATE)).reset_index(drop=True)

    print('Saving subsampled dataframe...')
    df_subs.to_parquet(args.output)
