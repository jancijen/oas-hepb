"""
Utility functions for plotting.

Author: Ing. David Prihoda
"""

from .criterions import translating_lm, smooth_masked_lm, weighted_sentence_prediction
from .tasks import translating_lm
from fairseq.models.roberta.model import base_architecture
from fairseq.models import register_model_architecture

@register_model_architecture('roberta', 'roberta_small')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    base_architecture(args)
