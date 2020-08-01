import argparse
import pandas as pd
from bin.parallel_execution import run_parallelly
from bin.build_clustered_data import CLUSTER_ID_COLUMN_NAME


def get_representative_sequences(cluster_data):
    cdr3_mode = cluster_data['cdr3'].mode().values[0]
    return cluster_data[cluster_data['cdr3'] == cdr3_mode]

def save_clusters_representative_sequences(clusters_data_path, out_path):
    print('Loading clusters data...')
    clusters_df = pd.read_parquet(clusters_data_path)

    print('Grouping data to by clusters...')
    clusters_gr = clusters_df.groupby(CLUSTER_ID_COLUMN_NAME)

    print('Computing clusters representative sequences...')
    params = [(cluster_data,) for _, cluster_data in clusters_gr]
    cluster_repr_sequences = run_parallelly(get_representative_sequences, params)

    print('Constructing output data...')
    merged_repr_data = pd.concat(cluster_repr_sequences, axis=0)

    print('Saving output data...')
    merged_repr_data.to_parquet(out_path)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Computes and saves sequences with the most frequently occuring CDR3 region in each cluster.')
    parser.add_argument('--clustered_data', action='store', help='Path to clustered sequences whose representative sequences should be computed.')
    parser.add_argument('--out_data', action='store', help='Path where dataframe with cluster representative sequences should be saved.')

    args = parser.parse_args()

    save_clusters_representative_sequences(args.clustered_data, args.out_data)
