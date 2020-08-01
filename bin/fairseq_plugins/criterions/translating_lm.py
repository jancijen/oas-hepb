# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)

# FIXME: this loss has some issues :(
#@register_criterion('translating_lm')
class TranslatingLMCriterion(FairseqCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits, _ = model(**sample['net_input'])
        targets = model.get_targets(sample, [logits])
        sample_size = sample['ntokens'] - sample['target'].size(0)  # number of tokens without eos
        
        src_tokens = sample['net_input']['src_tokens']
        
        #logger.info('INPUT EXAMPLE {}'.format(src_tokens[0]))
        #logger.info('OUTPUT EXAMPLE {}'.format(logits[0].detach().cpu().numpy().argmax(axis=1)))
        #logger.info('TARGET EXAMPLE {}'.format(targets[0]))
        #logger.info('-------')
        
        # Flatten all tensors
        src_tokens = src_tokens.view(-1)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            ignore_index=self.padding_idx,
            reduction='sum',
        )
        
        mask = targets != self.padding_idx
        masked_preds = logits[mask].argmax(dim=1)
        masked_targets = targets[mask]
        masked_inputs = src_tokens[mask]
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'ncorrect': utils.item((masked_preds == masked_targets).sum()),
            'input_correct': utils.item((masked_inputs == masked_targets).sum())
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / ntokens, nsentences, round=1)

        if len(logging_outputs) > 0 and 'input_correct' in logging_outputs[0]:
            input_correct = sum(log.get('input_correct', 0) for log in logging_outputs)
            metrics.log_scalar('input_accuracy', 100.0 * input_correct / ntokens, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
