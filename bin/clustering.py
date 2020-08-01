import multiprocessing
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy import stats


SEQ_CNT_COLUMN_NAME = 'Count'
CDR3_LENGTH_FOR_ALLOWED_MISMATCH = 12
CDR3_LEN_COL_LABEL = 'cdr3_len'


def preprocess_data(data):
    preprocessed = data[['v', 'j', 'cdr3']]
    preprocessed[CDR3_LEN_COL_LABEL] = preprocessed['cdr3'].str.len()
    
    return preprocessed

def np_apply_along_axis(func, axis, arr, kwargs):
    return np.apply_along_axis(func, axis, arr, **kwargs)

def apply_over_rows(func, data, **kwargs):
    """
    Applies given function over rows on given 2-D data in parallel.
    
    Args:
        func    function to be applied.
        data    2-D data on which function should be applied.
        kwargs  key-worded additional arguments to be passed to the function.

    Returns:
        output data after applying given function on the data.
    """

    axis = 1
    cpu_cnt = multiprocessing.cpu_count()
    
    chunks = [(func, axis, split_data, kwargs) for split_data in np.array_split(data, cpu_cnt) if split_data.size > 0]

    pool = multiprocessing.Pool()
    map_results = pool.starmap(np_apply_along_axis, chunks)
    
    pool.close()
    pool.join()

    return np.concatenate(map_results)

def mismatch_matrix_generator(seq, seq_matrix):
    return (seq != seq_matrix).sum(axis=1, dtype=np.uint8)

def cdr3_aa_matrix(group_data):
    seq_matrix = group_data['cdr3'].to_numpy(dtype=str)
    seq_matrix = seq_matrix.view('S4').reshape((seq_matrix.size, -1))
    return seq_matrix

def cdr3_cluster_matrices(group_data):
    seq_matrix = cdr3_aa_matrix(group_data)

    seq_len = seq_matrix.shape[1]
    allowed_mismatch_cnt = seq_len // CDR3_LENGTH_FOR_ALLOWED_MISMATCH + 1

    print(f'CDR3 sequences length: {seq_len}')
    print('Building mismatch and similarity matrices...')

    mismatch_matrix = apply_over_rows(mismatch_matrix_generator, seq_matrix, seq_matrix=seq_matrix)
    similarity_matrix = mismatch_matrix <= allowed_mismatch_cnt
    
    # Adjust mismatch matrix to show only mismatch counts under allowed limit
    mismatch_matrix[~similarity_matrix] = 0
    
    return similarity_matrix, mismatch_matrix

def cluster(data):
    print('Preprocessing data...')
    preprocessed_data = preprocess_data(data)
    grouped_data = preprocessed_data.groupby(['v', 'j', CDR3_LEN_COL_LABEL])
    
    print(f'Number of groups: {grouped_data.ngroups}')
    
    print('Clustering...')
    clusters = []
    clusters_cnt = 0
    TQDM_MESSAGE = 'Clustering ({} clusters)'
    tqdm_grouped_data = tqdm(grouped_data, desc=TQDM_MESSAGE.format(clusters_cnt))
    for _, group_data in tqdm_grouped_data:
        unclustered_seq_cnt = len(group_data)
        similarity_matrix, _ = cdr3_cluster_matrices(group_data)

        while unclustered_seq_cnt > 0:
            clusters_seq_cnts = similarity_matrix.sum(axis=0)
            center_seq_idx = np.argmax(clusters_seq_cnts)
            cluster_seq_indices = np.where(similarity_matrix[center_seq_idx])[0]

            similarity_matrix[cluster_seq_indices, :] = 0
            similarity_matrix[:, cluster_seq_indices] = 0

            orig_cluster_center_idx = group_data.iloc[center_seq_idx].name
            orig_cluster_indices = [group_data.iloc[sim_matrix_idx].name for sim_matrix_idx in cluster_seq_indices]
            clusters.append((orig_cluster_center_idx, orig_cluster_indices))

            unclustered_seq_cnt -= clusters_seq_cnts[center_seq_idx]

            clusters_cnt = clusters_cnt + 1
            tqdm_grouped_data.set_description(TQDM_MESSAGE.format(clusters_cnt))

    return clusters

def remove_rows_and_cols(matrix, indices_to_remove):
    matrix = np.delete(matrix, indices_to_remove, axis=0)
    matrix = np.delete(matrix, indices_to_remove, axis=1)
    
    return matrix

def cluster_group_data(group_data):
    unclustered_seq_cnt = len(group_data)
    print(f'Total number of sequences: {unclustered_seq_cnt}')

    similarity_matrix, mismatch_matrix = cdr3_cluster_matrices(group_data)
    
    # Keep collection of original indices to allow removal of rows and columns
    orig_indices = np.arange(unclustered_seq_cnt)

    clusters = []
    print('Clustering...')
    while unclustered_seq_cnt > 0:
        # Get points of the largest cluster
        clusters_seq_cnts = similarity_matrix.sum(axis=0)
        largest_cluster_size = np.max(clusters_seq_cnts)

        # All the remaining clusters are of size 1
        if largest_cluster_size <= 1:
            print(f'All remaining {unclustered_seq_cnt} sequences are in their own clusters (of size 1)...')
            orig_cluster_center_indices = list(group_data.iloc[orig_indices].index)

            clusters.extend([(orig_cluster_center_idx, [orig_cluster_center_idx]) for orig_cluster_center_idx in orig_cluster_center_indices])
            unclustered_seq_cnt -= len(orig_cluster_center_indices)
        else:
            center_seq_indices = np.where(clusters_seq_cnts == largest_cluster_size)[0]
            total_mismatches = mismatch_matrix[center_seq_indices].sum(axis=1)
            center_seq_idx = center_seq_indices[np.argmin(total_mismatches)]
        
            cluster_seq_indices = np.where(similarity_matrix[center_seq_idx])[0]

            # Get original indices of the points belonging to the cluster
            orig_cluster_center_idx = group_data.iloc[orig_indices[center_seq_idx]].name
            orig_cluster_indices = [group_data.iloc[orig_indices[sim_matrix_idx]].name for sim_matrix_idx in cluster_seq_indices]

            # "Save" constructed cluster
            clusters.append((orig_cluster_center_idx, orig_cluster_indices))
            unclustered_seq_cnt -= clusters_seq_cnts[center_seq_idx]

            # Remove the points assigne to the cluster and adjust collection of the original indices accordingly
            orig_indices = np.delete(orig_indices, cluster_seq_indices)
            similarity_matrix = remove_rows_and_cols(similarity_matrix, cluster_seq_indices)
            mismatch_matrix = remove_rows_and_cols(mismatch_matrix, cluster_seq_indices)

            print(f'New cluster size: {clusters_seq_cnts[center_seq_idx]}. Total clusters: {len(clusters)}. Unclustered sequences: {unclustered_seq_cnt}')

    return clusters

def get_cluster_consensus_seq(cluster_id, cluster_data):
    if len(cluster_data) <= 2:
        consensus_seq = cluster_data['cdr3'].iat[0]
    else:
        cdr3_aa = cdr3_aa_matrix(cluster_data)
        consensus_aa = stats.mode(cdr3_aa)[0]
        consensus_seq = consensus_aa.astype('|S1').tostring().decode('utf-8')

    return cluster_id, consensus_seq
