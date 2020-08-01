import argparse
import pandas as pd
from bin.clustering import cluster_group_data


CLUSTER_ID_COLUMN_NAME = 'Cluster_ID'
CLUSTER_CENTER_CDR3_COLUMN_NAME = 'Center_CDR3'


def save_clustered_data(df_path, out_path, out_centroids_path):
    """
    Clusters given dataframes and saves it.
    
    Args:
        df_path            paths to input dataframe.
        out_path           path where clustered data should be saved.
        out_centroids_path path where clusters centroid data should be saved.
    """

    print('Loading data...')
    clustered_seq_df = pd.read_parquet(df_path)

    print('Computing clusters for the data...')
    clusters = cluster_group_data(clustered_seq_df)

    print('Constructing output data...')
    cluster_centers_dfs = []
    for idx, (cluster_center_idx, datapoints_indices) in enumerate(clusters):
        # Clustered sequences
        clustered_seq_df.loc[datapoints_indices, CLUSTER_ID_COLUMN_NAME] = idx
        cluster_seqs_data = clustered_seq_df.loc[datapoints_indices]

        # Clusters centers
        cluster_center_cdr3 = clustered_seq_df.loc[cluster_center_idx]['cdr3']
        cluster_centers_dfs.append(cluster_seqs_data.loc[cluster_seqs_data['cdr3'] == cluster_center_cdr3])

    clustered_seq_df[CLUSTER_ID_COLUMN_NAME] = clustered_seq_df[CLUSTER_ID_COLUMN_NAME].astype(int)

    # Create clusters centers dataframe
    cluster_centers_df = pd.concat(cluster_centers_dfs)

    print('Saving output data...')
    clustered_seq_df.to_parquet(out_path)
    cluster_centers_df.to_parquet(out_centroids_path)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Constructs and saves clustered data.')
    parser.add_argument('--data', action='store', help='Paths to a dataframe which data should be clustered.')
    parser.add_argument('--out_data', action='store', help='Path where clustered dataframe should be saved.')
    parser.add_argument('--out_centroid_data', action='store', help='Path where cluster centroids dataframe should be saved.')

    args = parser.parse_args()

    save_clustered_data(args.data, args.out_data, args.out_centroid_data)
