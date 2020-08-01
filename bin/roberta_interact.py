"""
Functions intended for interaction with RoBERTa models
- predictions and feature extraction.
"""

import argparse
from fairseq.models.roberta import RobertaModel
import torch
import math
import numpy as np
import pandas as pd

from bin.roberta import seqs2tokens, seqs2features


def batches(X, batches_cnt):
    batch_size = math.ceil(X.shape[0] / batches_cnt)
    return [X[i:i+batch_size] for i in range(0, X.shape[0], batch_size)]

def crop_sequences(seqs, max_length):
    cropped_seqs = seqs.copy()
    long_sequences = cropped_seqs.loc[cropped_seqs.str.len() > max_length]
    cropped_seqs.loc[cropped_seqs.str.len() > max_length] = long_sequences.str.slice(stop=max_length)
    
    return cropped_seqs


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Make prediction or extract features for given data using given RoBERTa model.')
    parser.add_argument('--input', action='store')
    parser.add_argument('--seq_col', action='store')
    parser.add_argument('--max_len', action='store', default=None, type=int)
    parser.add_argument('--batch_cnt', action='store', type=int)
    parser.add_argument('--checkpoint_dir', action='store')
    parser.add_argument('--checkpoint_file', action='store')
    parser.add_argument('--data', action='store')
    parser.add_argument('--output', action='store')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--extract_features', action='store_true', default=False)

    args = parser.parse_args()

    print('Loading data...')
    input_data = pd.read_parquet(args.input)[args.seq_col]

    if args.max_len:
        print(f'Cropping data to length {args.max_len}...')
        input_data = crop_sequences(input_data, args.max_len)

    print('Loading RoBERTa model...')
    roberta = RobertaModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data,
        user_dir='bin/fairseq_plugins/',
        bpe=None
    )
    roberta.eval() # disable dropout

    if not args.cpu:
        roberta.cuda()

    print('Pre-processing data...')
    input_tokens = seqs2tokens(input_data, roberta)

    if not args.cpu:
        input_tokens.cuda()

    X_batches = batches(input_tokens, args.batch_cnt)

    if args.extract_features:
        # Extract features
        print('Generating features...')
        features = []

        for i, X_batch in enumerate(X_batches):
            print(f'Generating features for batch #{i} of shape {X_batch.shape}...')
            features.extend(seqs2features(X_batch, roberta, 'cls', args.cpu))

        print('Saving features...')
        features_df = pd.DataFrame(features, index=input_data.index)
        features_df.columns = [str(col_name) for col_name in features_df.columns]
        features_df.to_parquet(args.output)
    else:
        # Make predictions
        preds = []

        with torch.no_grad():
            for i, X_batch in enumerate(X_batches):
                print(f'Predicting targets for batch #{i} of shape {X_batch.shape}...')
                preds.extend(roberta.predict('sentence_classification_head', X_batch).tolist())
        
        print('Saving predictions...')
        np.save(args.output, preds)
