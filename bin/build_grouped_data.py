import pathlib
import argparse
import pandas as pd

GROUP_FILES_LIST_FILENAME = 'groups-list.txt'
CDR3_LEN_COL_LABEL = 'cdr3_len'
FILENAME_CHAR_REPLACEMENTS = {
    '*': 'x'
}


def save_grouped_data(input_df, output_dir):
    """
    Creates dataframes grouped by V, J segments and CDR3 length and saves them.

    Args:
        input_df    path to a dataframe whose grouped dataframes should be built.
        output_dir  path to a directory where output data should be saved.
    """

    df = pd.read_parquet(input_df)
    df[CDR3_LEN_COL_LABEL] = df['cdr3'].str.len()

    grouped_df = df.groupby(['v', 'j', CDR3_LEN_COL_LABEL])

    if grouped_df:
        path_dir = pathlib.Path(output_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for (v, j, cdr3_len), grouped_data in grouped_df:
        filename = f'{v}_{j}_{cdr3_len}'

        for char, replacement_char in FILENAME_CHAR_REPLACEMENTS.items():
            filename = filename.replace(char, replacement_char)
        
        filenames.append(filename)
        grouped_data.to_parquet(f'{output_dir}/{filename}.parquet', index=False)

    with open(f'{output_dir}/{GROUP_FILES_LIST_FILENAME}', 'w') as groups_list_file:
        groups_list_file.write('\n'.join(filenames))


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Build data grouped by V, J segments and CDR3 length.')
    parser.add_argument('--data', action='store', help='Path to a dataframe whose grouped dataframes should be built.')
    parser.add_argument('--out_dir', action='store', help='Path to a directory where output data should be saved.')

    args = parser.parse_args()

    save_grouped_data(args.data, args.out_dir)
