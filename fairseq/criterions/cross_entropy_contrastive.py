# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_contrastive", dataclass=CrossEntropyCriterionConfig)
class CrossEntropy_Contrastive_Criterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temperature = 0.1
        self.contrastive_lamda = 0
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # cross entropy on decoder
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        # cross entropy for nmt
        net_output_mt = model.mt_forward(**sample["net_input"])
        loss_mt, _ = self.compute_loss(model, net_output_mt, sample, reduce=reduce)

        loss_decoder = 1.0 * loss #+ 0.1 * loss_mt
        # contrastive loss on encoder
        src_tokens, src_lengths = sample["net_input"]["source"], sample["net_input"]["source_lengths"]
        text_encoder_out = model.encoder.context_encoder.forward(src_tokens, src_lengths)
        text_feature = text_encoder_out["encoder_out"][0] # T x B x C
        text_padding_mask = text_encoder_out["encoder_padding_mask"][0]

        image_encoder_out = model.encoder.forward(sample["net_input"]["img_source"])
        image_features = image_encoder_out["encoder_out"][0] # T x B x C
        image_padding_mask = torch.ones(image_features.shape[1], image_features.shape[0], device=image_features.device)
        contrastive_loss = self.get_contrastive_loss(
            text_feature,
            image_features,
            1-text_padding_mask.float(),
            image_padding_mask)
        # ) + self.get_contrastive_loss(
        #     image_features,
        #     image_features,
        #     image_padding_mask,
        #     image_padding_mask,
        # ) ) / 2

        all_loss = loss_decoder + self.contrastive_lamda * contrastive_loss * sample["ntokens"] / sample["target"].size(0)
        # contrastive loss
        
        # bz = image_sentence_representation.shape[0]
        # temperature = 0.2
        # sim = self.cos(image_sentence_representation.repeat([1,bz]).reshape(bz**2,-1).float(), text_sentence_representation.repeat([bz,1]).float())
        # logits = torch.exp(sim.view(bz,bz) / temperature)

        # contrastive_loss = - torch.log(torch.diag(logits) / logits.sum(dim=-1)).sum()

        
        logging_output = {
            "loss": loss.data,
            "loss_mt": loss_mt.data,
            "ctr_loss": contrastive_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return all_loss, sample_size, logging_output

    def similarity_function(self, ):
        return self.cos
    def get_contrastive_loss(self, encoder_out1, encoder_out2, mask1, mask2):
        
        def _sentence_embedding(encoder_out, mask):
            encoder_output = encoder_out.transpose(0, 1) # B x T x C
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding
        
        encoder_embedding1 = _sentence_embedding(encoder_out1, mask1)  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(encoder_out2, mask2)  # [batch, hidden_size]
        
        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2
        
        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
        
        loss = -torch.nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        
        return loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ctr_loss_sum = sum(log.get("ctr_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("loss_mt", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs) 
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mt", mt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctr_loss", ctr_loss_sum / nsentences, sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
