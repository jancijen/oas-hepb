import argparse
import pandas as pd


def save_concatenated_frame(df_paths, out_path):
    """
    Concatenates given dataframes and saves it.
    
    Args:
        df_paths    paths to input dataframes.
        out_path    path where concatenated dataframe should be saved.
    """

    dfs = [pd.read_parquet(df_path) for df_path in df_paths]
    concatenated_df = pd.concat(dfs, axis=0)
    concatenated_df.to_parquet(out_path, compression='gzip', engine='pyarrow')


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Constructs and saves a concatenated dataframe.')
    parser.add_argument('--data', action='store', nargs='+', help='Paths of dataframes which should be concatenated.')
    parser.add_argument('--out_data', action='store', help='Path where concatenated dataframe should be saved.')

    args = parser.parse_args()

    save_concatenated_frame(args.data, args.out_data)