# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import time
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules import (
    GumbelVectorQuantizer,
)
from fairseq.modules import PositionalEmbedding
import easyocr
import logging
logger = logging.getLogger(__name__)

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class ItransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens):
        self.cfg = cfg
        super().__init__(dictionary)
        self.max_source_positions = cfg.max_source_positions
        
        
        # feature encoder (VGG)
        if cfg.use_pretrained_ocr:
            logger.info("Using pretrained ocr model... ")
            from .vgg_model import Model
            self.cnn_encoder = Model(input_channel=1, output_channel=256, hidden_size=256, multi_line=self.cfg.multi_line)
            # load pretrained model
            model_path = cfg.pretrained_ocr_path
            state_dict = torch.load(model_path, map_location="cpu")
            new_state_dict = self.cnn_encoder.state_dict()
            for key, value in state_dict.items():
                new_key = key[7:]
                if new_key in new_state_dict:
                    new_state_dict[new_key] = value
            self.cnn_encoder.load_state_dict(new_state_dict)

            # subsample to shrink sequence length
            self.subsample = Conv1dSubsampler(256, 256, cfg.encoder.embed_dim, kernel_sizes=(5,5))
            
        else:
            # resnet
            from .resnet import Model
            self.cnn_encoder = Model(self.cfg)
            for m in self.cnn_encoder.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight, gain=1)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                
            self.subsample = None

        
        self.use_codebook = cfg.use_codebook
        if self.use_codebook:
            self.codebook_prob = cfg.codebook_prob
            vq_dim = cfg.encoder.embed_dim
            input_dim = cfg.encoder.embed_dim
            temperature = eval(cfg.vq_temp)
            self.quantizer = GumbelVectorQuantizer(
                dim=input_dim,
                num_vars=cfg.vq_num_vars,
                temp=temperature,
                groups=cfg.vq_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.vq_quantizer_depth,
                weight_proj_factor=cfg.vq_quantizer_factor,
            )
        
        
        from fairseq.modules import LayerNorm
        self.layernorm = LayerNorm(cfg.encoder.embed_dim)
        

        
        # context encoder (Transformer)
        from fairseq.models.transformer import TransformerEncoderBase
        self.context_encoder = TransformerEncoderBase(cfg, dictionary, embed_tokens)

        
            
        
        # easyocr model
        # self.reader = easyocr.Reader(['ch_sim','en'],gpu=False,detector=False,recognizer=True,quantize=False)
        # self.recognizer = self.reader.recognizer
        # self.recognizer.train()
        # for name, params in self.recognizer.named_parameters():
        #     params.requires_grad = True

        # from .craft import CRAFT, copyStateDict
        # self.detector = CRAFT()
        # trained_model = "/home/lsj/.EasyOCR/model/craft_mlt_25k.pth"
        # self.detector.load_state_dict(copyStateDict(torch.load(trained_model, map_location="cpu")))

        # self.detector.eval()
        # for name, params in self.detector.named_parameters():
        #     params.requires_grad = False
        

    
    

    def forward(
        self,
        img_source,
        **kwargs,
    ):
        return self.forward_scriptable(img_source)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(self, img_source):
        
        inp = img_source
        x = self.cnn_encoder.extract_features(inp) # (B, W, C) / B x T x C
        bz, time_step, _ = x.shape
        
        # extra cnn module to reduce sequence length
        src_lengths = torch.ones(bz, device=x.device).int() * time_step
        if self.subsample is not None:
            x, src_lengths = self.subsample(x, src_lengths)
            bz, time_step, _ = x.shape
        
        x = self.layernorm(x) # B x T x C

        dummy_tokens = torch.zeros((bz, src_lengths[0]), device=x.device)
        x = self.context_encoder(dummy_tokens, src_lengths, token_embeddings=x) 

        # vector quantization
        if self.use_codebook and self.training:
            q_input = x["encoder_out"][0].transpose(0,1) # B x T x C
            q = self.quantizer(q_input)

            # q["x"]: B x T x C
            random_idx = torch.randperm(q["x"].size(1))[:int(q["x"].size(1) * self.codebook_prob)]
            # Make weight for q
            q_w = q["x"].new_zeros(q["x"].size(1))
            q_w[random_idx] = 1.0
            # Combine quantized codes and encoder output
            q_output = q_w.view(-1, 1) * q["x"] + (- q_w + 1).view(-1, 1) * q_input

            code_ppl = q["code_perplexity"] * bz * time_step
            prob_ppl = q["prob_perplexity"] * bz * time_step
        
            x["encoder_out"][0] = q_output.transpose(0,1) # T x B x C

        if self.use_codebook and self.training:
            x["code_perplexity"] = code_ppl
            x["prob_perplexity"] = prob_ppl
        return x

        # return {
        #     "encoder_out": [x.transpose(0,1)],  # T x B x C
        #     "encoder_padding_mask": [torch.zeros((bz, time_step), device=x.device)],  # B x T
        #     "src_lengths": [src_lengths],
        # }
    
    # def forward(
    #     self,
    #     src_tokens,
    #     src_lengths: Optional[torch.Tensor] = None,
    #     return_all_hiddens: bool = False,
    #     token_embeddings: Optional[torch.Tensor] = None,
    #     **kwargs
    #     ):
    #     return self.context_encoder(src_tokens, src_lengths)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]


        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_padding_mask = []
        else:
            new_padding_mask = [(encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_padding_mask,
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        pass


