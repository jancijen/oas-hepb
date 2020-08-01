import numpy as np
import math

def batches(X, batches_cnt):
    batch_size = math.ceil(X.shape[0] / batches_cnt)
    return [X[i:i+batch_size] for i in range(0, X.shape[0], batch_size)]

def iterate_batches(X, y, batches_cnt):
    X_batches = batches(X, batches_cnt)
    y_batches = batches(y, batches_cnt)

    for X_batch, y_batch in zip(X_batches, y_batches):
        yield X_batch, y_batch

def predict_in_batches(model, X, batches_cnt=10, probabilities=False, verbose=False):
    X_batches = batches(X, batches_cnt)
    
    pred = []
    for i, X_batch in enumerate(X_batches):
        if verbose:
            print(f'Predicting targets for batch #{i} of shape {X_batch.shape}...')
        if probabilities:
            pred.extend(model.predict_proba(X_batch))
        else:
            pred.extend(model.predict(X_batch))
        
    return pred