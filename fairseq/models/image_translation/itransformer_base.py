# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.image_translation import ItransformerConfig, ItransformerEncoderBase
from fairseq.models.transformer import TransformerDecoderBase


logger = logging.getLogger(__name__)


class ItransformerModelBase(FairseqEncoderDecoderModel):
    """

    Args:
        encoder (CnnEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, ItransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        cfg.encoder.cnn_layers = eval(cfg.encoder.cnn_layers)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        if cfg.cnnfix:
            for params in encoder.cnn_encoder.parameters():
                params.requires_grad = False
        if cfg.embfix:
            for params in decoder.parameters():
                params.requires_grad = False
            for i in range(len(decoder.layers)):
                decoder.layers[i].encoder_attn.q_proj.weight.requires_grad = True
                decoder.layers[i].encoder_attn.q_proj.bias.requires_grad = True
                decoder.layers[i].encoder_attn.k_proj.weight.requires_grad = True
                decoder.layers[i].encoder_attn.k_proj.bias.requires_grad = True
                decoder.layers[i].encoder_attn.v_proj.weight.requires_grad = True
                decoder.layers[i].encoder_attn.v_proj.bias.requires_grad = True
                decoder.layers[i].encoder_attn.out_proj.weight.requires_grad = True
                decoder.layers[i].encoder_attn.out_proj.bias.requires_grad = True

            encoder.context_encoder.embed_tokens.weight.requires_grad = False

        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):

        return ItransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        
        model_state_dict = self.state_dict()
        initialized_keys = set([])
        initialized_num = 0
        total_num = 0
        random_key = []
        ignore_key = []
        removed_key = []
        for key in state_dict:
            
            if key in ignore_key:
                continue
            if key in model_state_dict:
                model_state_dict[key] = state_dict[key].to(model_state_dict[key].device)
                initialized_keys.add(key)
                initialized_num += state_dict[key].numel()
            else:
                # try
                splited = key.split(".")
                new_key = '.'.join(splited[0:1] + ["context_encoder"] + splited[1:])
                if new_key in model_state_dict:
                    model_state_dict[new_key] = state_dict[key].to(model_state_dict[new_key].device)
                    initialized_keys.add(new_key)
                    initialized_num += state_dict[key].numel()
                else:
                    removed_key.append(key)


        for key in model_state_dict:
            total_num += model_state_dict[key].numel()
            if key not in initialized_keys:
                random_key.append(key)

        logger.info(f"Keys initialized with pretrained model: {initialized_keys}")
        logger.info(f"Keys randomly initialized : {random_key}")
        logger.info(f"Keys removed from pretrained model: {removed_key}")
        logger.info(f"Parameters coverage percent: {initialized_num / total_num}")
                    
            
        return super().load_state_dict(model_state_dict, strict, model_cfg, args)
    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        img_source,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(img_source)
        src_lengths = encoder_out["src_lengths"]
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out[1]["encoder_out"] = encoder_out
        if self.cfg.use_codebook and self.training:
            decoder_out[1]["code_perplexity"] = encoder_out["code_perplexity"]
            decoder_out[1]["prob_perplexity"] = encoder_out["prob_perplexity"]
        
        return decoder_out
    def mt_forward(self,
            source_text,
            source_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            **kwargs):
        encoder_out = self.encoder.context_encoder(source_text, source_lengths)
        src_lengths = encoder_out["src_lengths"]
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out[1]["encoder_out"] = encoder_out
        return decoder_out
    def extract_text_features(self, src_tokens, src_lengths):
        return self.encoder.context_encoder(src_tokens, src_lengths)
    def extract_image_features(self, img_source):
        encoder_out = self.encoder(img_source)
        return encoder_out
    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
