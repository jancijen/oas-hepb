# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    AssertSameLengthDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils


logger = logging.getLogger(__name__)


@register_task('translating_lm')
class TranslatingLMTask(FairseqTask):
    """Task for training same-length translation models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='data directory, containing input and label directories')
        parser.add_argument('--no-shuffle', action='store_true', default=False)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        source_dictionary = Dictionary.load(os.path.join(paths[0], 'input', 'dict.txt'))
        source_dictionary.add_symbol('<mask>')
        logger.info('dictionary: {} types'.format(len(source_dictionary)))
        target_dictionary = Dictionary.load(os.path.join(paths[0], 'label', 'dict.txt'))
        target_dictionary.add_symbol('<mask>')
        assert source_dictionary == target_dictionary, 'Target dictionary should be equal to src dictionary'
        return cls(args, source_dictionary)
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        
        src_tokens = sample['net_input']['src_tokens'].view(-1)

        return loss, sample_size, logging_output

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            assert dataset is not None, 'could not find dataset: {}'.format(get_path(type, split))
            return dataset

        src_tokens = make_dataset('input', self.dictionary)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        tgt_tokens = make_dataset('label', self.dictionary)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'target': RightPadDataset(
                    tgt_tokens,
                    pad_idx=self.dictionary.pad(),
                ),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
            '_assert_lengths_match': AssertSameLengthDataset(src_tokens, tgt_tokens),
        }

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
    
@register_task('paired_lm') # TODO remove
class PairedLMTask(TranslatingLMTask):
    pass