# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from sklearn.metrics import confusion_matrix

@register_criterion('weighted_sentence_prediction')
class WeightedSentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target, freeze_encoder, class_weights):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target
        self.freeze_encoder = freeze_encoder
        self.class_weights = torch.cuda.FloatTensor(class_weights)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--freeze-encoder', action='store_true', default=False,
                            help='Freeze encoder weights and disable encoder dropout during training')
        parser.add_argument('--class-weights', action='store', nargs='*', default=[], type=float)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
            freeze_encoder=self.freeze_encoder,
            class_weights=self.class_weights
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
#             if torch.cuda.is_available():
#                 self.class_weights.cuda()
            loss = F.nll_loss(lprobs, targets, weight=self.class_weights, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            'neg_class_weight': self.class_weights[0].item(),
            'pos_class_weight': self.class_weights[1].item()
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)

            tn, fp, fn, tp = confusion_matrix(targets.cpu(), preds.cpu(), labels=[0, 1]).ravel()
            logging_output['tn'] = tn
            logging_output['tp'] = tp
            logging_output['fn'] = fn
            logging_output['fp'] = fp

            logging_output['ncorrect'] = (preds == targets).sum()

        weighted_sample_size = ((tp + fn) * self.class_weights[1].item()) + ((tn + fp) * self.class_weights[0].item())

        return loss, weighted_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        tn = sum(log.get('tn', 0) for log in logging_outputs)
        tp = sum(log.get('tp', 0) for log in logging_outputs)
        fn = sum(log.get('fn', 0) for log in logging_outputs)
        fp = sum(log.get('fp', 0) for log in logging_outputs)

        pos_cl_weight = logging_outputs[0].get('pos_class_weight')
        neg_cl_weight = logging_outputs[0].get('neg_class_weight')

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * (tp * pos_cl_weight + tn * neg_cl_weight) / (tp * pos_cl_weight + tn * neg_cl_weight + fp * pos_cl_weight + fn * neg_cl_weight), nsentences, round=1)

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            metrics.log_scalar('precision', 100.0 * precision, nsentences, round=1)
            metrics.log_scalar('recall', 100.0 * recall, nsentences, round=1)
            metrics.log_scalar('F1', 100.0 * f1, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True