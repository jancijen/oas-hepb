from parallel_execution import run_parallelly

import argparse
import pandas as pd
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


ALIGNMENT_MATRIX = matlist.blosum62
OPEN_GAP_PENALIZATION = -5
EXTEND_GAP_PENALIZATION = -2


def get_best_alignment(seq, target_seqs):
    if seq in target_seqs.values:
        return seq, seq, None
        
    alignments = target_seqs.apply(lambda target_seq: pairwise2.align.globalds(seq, target_seq, ALIGNMENT_MATRIX, OPEN_GAP_PENALIZATION, EXTEND_GAP_PENALIZATION, one_alignment_only=True)[0])
    best_alignment = max(alignments, key=lambda al: al.score)

    return best_alignment[:3]


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Aligns sequences and saves the best alignments.')
    parser.add_argument('--source', action='store', help='Path to a source dataframe with sequences.')
    parser.add_argument('--target', action='store', help='Path to a target dataframe with sequences.')
    parser.add_argument('--seq_col', action='store', help='Name of a column with sequences.')
    parser.add_argument('--out_data', action='store', help='Path where aligned sequences should be saved.')

    args = parser.parse_args()

    print('Loading input data...')
    source_seqs = pd.read_parquet(args.source)[args.seq_col]
    target_seqs = pd.read_parquet(args.target)[args.seq_col]

    print('Finding the best alignments...')
    params = [(source_seq, target_seqs) for source_seq in source_seqs]
    alignments = run_parallelly(get_best_alignment, params)

    print('Saving output data...')
    pd.DataFrame(alignments, columns=['Source_Seq', 'Target_Seq', 'Score']).to_parquet(args.out_data)
