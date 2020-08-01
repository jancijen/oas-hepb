import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from bin.build_clustered_data import CLUSTER_ID_COLUMN_NAME


RANDOM_STATE = 42
TEST_SIZE = 0.33


def features_and_targets(X_dfs, y):
    '''
    Constructs single feature frame and corresponding targets.
    '''
    
    X_subs = shuffle(pd.concat(X_dfs), random_state=RANDOM_STATE)
    y_sub = y.loc[X_subs.index]
    
    return X_subs, y_sub

def save_subsampled_data(X_train, X_valid, y_train, y_valid, cluster_sizes, pos_fraction, X_train_data, X_valid_data, y_train_data, y_valid_data):
    y_val_coutns = y_train.value_counts()

    if pos_fraction < 1:
        print('Subsampling positive data...')
        X_pos = X_train.sample(int(y_val_coutns[True] * pos_fraction), weights=cluster_sizes.loc[y_train.index[y_train]], random_state=RANDOM_STATE)
    else:
        X_pos = X_train.loc[y_train]

    # Subsample negatives to match positive count
    print('Subsampling negative data...')
    X_neg = X_train.sample(len(X_pos), weights=cluster_sizes.loc[y_train.index[y_train == False]], random_state=RANDOM_STATE)

    print(f'Positive training data shape: {X_pos.shape}')
    print(f'Negative training data shape: {X_neg.shape}')

    X_train, y_train = features_and_targets([X_pos, X_neg], y_train)

    print(f'Trainining data shape: {X_train.shape}, target: {y_train.shape}')
    print(f'Validation data shape: {X_valid.shape}, target: {y_valid.shape}')

    # Save resulting data
    print('Saving training data...')
    X_train.to_parquet(X_train_data)
    pd.DataFrame(y_train, columns=['HepB']).to_parquet(y_train_data)

    print('Saving validation data...')
    X_valid.to_parquet(X_valid_data)
    pd.DataFrame(y_valid, columns=['HepB']).to_parquet(y_valid_data)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Splits given data to train and validation data and saves it.')
    # Input args
    parser.add_argument('--X_data', action='store', help='Path from where X data should be loaded.')
    parser.add_argument('--y_data', action='store', help='Path from where y data should be loaded.')
    parser.add_argument('--clustered_data', action='store', help='Path from where clustered data should be loaded.')
    parser.add_argument('--pos_fraction', action='store', type=float, default=1, help='Fraction of positive training datapoints that should be subsampled.')
#     parser.add_argument('--neg_fraction', action='store', type=float, default=1, help='Fraction of negative training datapoints that should be subsampled.')
    # Output args
    parser.add_argument('--X_train_data', action='store', help='Path where training features dataframe should be saved.')
    parser.add_argument('--X_valid_data', action='store', help='Path where validation features dataframe should be saved.')
    parser.add_argument('--y_train_data', action='store', help='Path where training targets dataframe should be saved.')
    parser.add_argument('--y_valid_data', action='store', help='Path where validation targets dataframe should be saved.')

    args = parser.parse_args()

    # Load data
    print('Loading data...')
    X = pd.read_parquet(args.X_data)
    y = pd.read_parquet(args.y_data).iloc[:,0]

    # Split data
    print('Splitting data...')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # (Optional) Positive subsampling
    cluster_sizes = pd.read_parquet(args.clustered_data).groupby(CLUSTER_ID_COLUMN_NAME).size()
    save_subsampled_data(X_train, X_valid, y_train, y_valid, cluster_sizes, args.pos_fraction, args.X_train_data, args.X_valid_data, args.y_train_data, args.y_valid_data)