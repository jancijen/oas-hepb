import argparse
import pandas as pd
from bin.clustering import get_cluster_consensus_seq
from bin.parallel_execution import run_parallelly
from bin.build_clustered_data import CLUSTER_ID_COLUMN_NAME


OUTPUT_CDR3_CONSENSUS_SEQ_COLUMN_NAME = 'CDR3 Consensus Sequence'


def save_clusters_consensus_sequences(df_path, out_path):
    print('Loading data...')
    df = pd.read_parquet(df_path)

    print('Grouping data to by clusters...')
    clusters_gr = df.groupby(CLUSTER_ID_COLUMN_NAME)

    print('Computing clusters consensus sequences...')
    params = [cluster_params for cluster_params in clusters_gr]
    cluster_consensus_sequences = run_parallelly(get_cluster_consensus_seq, params)

    print('Constructing output data...')
    res_df = pd.DataFrame(cluster_consensus_sequences, columns=[CLUSTER_ID_COLUMN_NAME, OUTPUT_CDR3_CONSENSUS_SEQ_COLUMN_NAME])
    res_df = res_df.set_index(CLUSTER_ID_COLUMN_NAME)

    print('Saving output data...')
    res_df.to_parquet(out_path)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Computes and saves consensus sequences of each cluster.')
    parser.add_argument('--data', action='store', help='Paths to a clustered data whose consensus sequences should be computed.')
    parser.add_argument('--out_data', action='store', help='Path where dataframe with cluster consensus sequences should be saved.')

    args = parser.parse_args()

    save_clusters_consensus_sequences(args.data, args.out_data)
