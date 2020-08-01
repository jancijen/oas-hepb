"""
Utility functions for RoBERTa models.

Author: Ing. David Prihoda
"""

import torch
import torch.nn as nn
import swifter
from tqdm import tqdm
import numpy as np
from fairseq import utils

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    if hasattr(lst, '__getitem__'):
        # Use slicing
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    # Use naive iteration        
    batch = []
    for val in lst:
        batch.append(val)
        if len(batch) == n:
            yield batch
            batch = []
    if len(batch):
        yield(batch)

def seq2sentence(seq):
    return ' '.join(list(seq))

def seq2tokens(dictionary, seq):
    assert ' ' not in seq, f'Expected regular sequence without space separators, found space in "{seq}"'
    sentence = '<s> ' + seq2sentence(seq).replace('*','<mask>') + ' </s>'
    return dictionary.encode_line(sentence, append_eos=False, add_if_not_exist=False).long()

def seqs2tokens(seqs, roberta, pad=True):
    dictionary = roberta.task.source_dictionary
    tokens = seqs.reset_index(drop=True).swifter.apply(lambda seq: seq2tokens(dictionary, seq))
    if not pad:
        return tokens
    return nn.utils.rnn.pad_sequence(
        tokens,
        batch_first=True, 
        padding_value=dictionary.pad_index
    )

def seqs2features(tokens, roberta, mode, cpu=False):
    if not cpu:
        tokens.cuda()
    with utils.eval(roberta.model):
        with torch.no_grad():
            features, extra = roberta.model(tokens, features_only=True, return_all_hiddens=False, requires_grad=False)
            features = features.detach().numpy()
            # mask out positions with padding
            features[tokens==roberta.task.source_dictionary.pad_index] = np.nan
            # mask out EOS
            features[tokens==roberta.task.source_dictionary.eos_index] = np.nan
    if mode == 'average':
        return np.nanmean(features, axis=1)
    elif mode == 'cls':
        return features[:, 0, :]
    elif mode == 'positional':
        return features[:, 1:-1, :]
    else:
        raise NotImplementedError(mode)
