# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from collections import OrderedDict
import torch.nn.functional as F
@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

    



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_contrastive", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropy_Contrastive_Criterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.use_contrastive_learning = task.cfg.use_contrastive_learning
        self.use_knowledge_distillation = task.cfg.use_knowledge_distillation
        self.use_multi_task_learning = task.cfg.use_multi_task_learning
        self.use_jsd = task.cfg.use_jsd

        self.ctr_lamda = task.cfg.ctr_lamda
        self.kd_lamda = task.cfg.kd_lamda
        self.mtl_it_lamda = task.cfg.mtl_it_lamda
        self.mtl_mt_lamda = task.cfg.mtl_mt_lamda
        self.jsd_lamda = task.cfg.jsd_lamda
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temperature = 0.1

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # it task
        it_sample, ext_mt_sample = None, None
        if isinstance(sample, OrderedDict):
            it_sample = sample["it"]
            ext_mt_sample = sample["mt"]
        else:
            it_sample = sample
        
        net_output = model(**it_sample["net_input"])
        # cross entropy on decoder
        loss_it, nll_loss_it, lprobs_it, target_it = self.compute_loss(model, net_output, it_sample, reduce=reduce)
        sample_size = (
            it_sample["target"].size(0) if self.sentence_avg else it_sample["ntokens"]
        )

        logging_output = {
            "loss": loss_it.data,
            "nll_loss":nll_loss_it.data,
            "ntokens": it_sample["ntokens"],
            "nsentences": it_sample["target"].size(0),
            "sample_size": sample_size,
        }

        # cross entropy for nmt
        loss_mt, loss_kd, loss_jsd = 0, 0, 0
        net_output_mt = None
        if self.use_multi_task_learning and self.mtl_mt_lamda > 0:
            net_output_mt = model.mt_forward(**it_sample["net_input"])
            loss_mt, nll_loss_mt, lprobs_mt, target_mt = self.compute_loss(model, net_output_mt, it_sample, reduce=reduce)

            logging_output["loss_mt"] = loss_mt.data
            logging_output["nll_loss_mt"] = nll_loss_mt.data

            if self.use_jsd and self.jsd_lamda > 0:
                loss_jsd = self.compute_jsd_loss(lprobs_it, lprobs_mt, target_it, target_mt, self.padding_idx)
                logging_output["loss_jsd"] = loss_jsd.data
                


        if self.use_knowledge_distillation and self.kd_lamda > 0:
            # knowledge distillation
            # if load fix text model
            if self.task.text_model is not None:
                src_tokens, src_lengths, prev_output_tokens = it_sample["net_input"]["source_text"], it_sample["net_input"]["source_lengths"], it_sample["net_input"]["prev_output_tokens"]
                self.task.text_model = self.task.text_model.to(loss_it.device)
                net_output_mt = self.task.text_model(src_tokens, src_lengths, prev_output_tokens)
            # else use recent It model
            elif net_output_mt is None:
                net_output_mt = model.mt_forward(**it_sample["net_input"])
            
            loss_kd = self.compute_kd_loss(model, net_output, net_output_mt, reduce=reduce)
            logging_output["loss_kd"] = loss_kd.data
        
        loss = self.mtl_it_lamda * loss_it + self.mtl_mt_lamda * loss_mt + self.kd_lamda * loss_kd + self.jsd_lamda * loss_jsd

        # contrastive loss on encoder
        if self.use_contrastive_learning and self.ctr_lamda > 0:
            # if load fix text model
            if self.task.text_model is not None:
                src_tokens, src_lengths = it_sample["net_input"]["source_text"], it_sample["net_input"]["source_lengths"]
                self.task.text_model = self.task.text_model.to(loss.device)
                text_encoder_out = self.task.text_model.extract_text_features(src_tokens, src_lengths)
            elif net_output_mt is not None:
                text_encoder_out = net_output_mt[1]["encoder_out"]
            else:
                # else use recent It model
                src_tokens, src_lengths = it_sample["net_input"]["source_text"], it_sample["net_input"]["source_lengths"]
                text_encoder_out = model.extract_text_features(src_tokens, src_lengths)
    

            text_feature = text_encoder_out["encoder_out"][0].detach() # T x B x C
            text_padding_mask = text_encoder_out["encoder_padding_mask"][0]

            #image_encoder_out = model.encoder.forward(sample["net_input"]["img_source"])
            image_encoder_out = net_output[1]["encoder_out"]
            image_features = image_encoder_out["encoder_out"][0] # T x B x C
            image_padding_mask = torch.ones(image_features.shape[1], image_features.shape[0], device=image_features.device)
            contrastive_loss = self.get_contrastive_loss(
                text_feature,
                image_features,
                1-text_padding_mask.float(),
                image_padding_mask)

            logging_output["ctr_loss"] = contrastive_loss.data
            
            loss = loss + self.ctr_lamda * contrastive_loss * it_sample["ntokens"] / it_sample["target"].size(0)

        
        if model.cfg.use_codebook and model.training:
            logging_output["code_perplexity"] = net_output[1]["code_perplexity"].data
            logging_output["prob_perplexity"] = net_output[1]["prob_perplexity"].data
 
        if ext_mt_sample is not None:
            # externel mt task
            ext_mt_sample["net_input"]["source_text"] = ext_mt_sample["net_input"]["src_tokens"]
            ext_mt_sample["net_input"]["source_lengths"] = ext_mt_sample["net_input"]["src_lengths"]
            # cross entropy for nmt
            net_output = model.mt_forward(**ext_mt_sample["net_input"])
            ext_loss, ext_nll_loss, _, _ = self.compute_loss(model, net_output, ext_mt_sample, reduce=reduce)
            sample_size_mt_ext = (
                ext_mt_sample["target"].size(0) if self.sentence_avg else ext_mt_sample["ntokens"]
            )

            loss = loss + ext_loss * it_sample["ntokens"] / ext_mt_sample["ntokens"]

            
            ext_logging_output = {
            "loss_mt_ext": ext_loss.data,
            "nll_loss_mt_ext": ext_nll_loss.data,
            "ntokens_mt_ext": ext_mt_sample["ntokens"],
            "nsentences_mt_ext": ext_mt_sample["target"].size(0),
            "sample_size_mt_ext": sample_size_mt_ext,
            }

            for key in ext_logging_output:
                logging_output[key] = ext_logging_output[key]


        
            
        if self.report_accuracy and model.training:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return loss, sample_size, logging_output
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

    def get_lprobs(self, model, net_output):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1))

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_kd_loss(self, model, student_output, teacher_output, reduce=True):
        slprobs = self.get_lprobs(model, student_output) # B*T x V
        tlprobs = self.get_lprobs(model, teacher_output).detach()
        if reduce:
            kd_loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)(slprobs, tlprobs)
        else:
            kd_loss = torch.nn.KLDivLoss(reduction='none', log_target=True)(slprobs, tlprobs)

        return kd_loss
        
    def compute_jsd_loss(self, lprobs_st, lprobs_mix, target_st, target_mix, ignore_index):
        kl_loss_st = F.kl_div(lprobs_mix, lprobs_st, log_target=True, reduction="none").sum(-1)
        kl_loss_mix = F.kl_div(lprobs_st, lprobs_mix, log_target=True, reduction="none").sum(-1)
        pad_mask = target_st.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = target_mix.eq(ignore_index)
        kl_loss_mix.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mix = kl_loss_mix.sum()
        kl_loss = (kl_loss_st + kl_loss_mix) / 2.0
        return kl_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ctr_loss_sum = sum(log.get("ctr_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("loss_mt", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("nll_loss_mt", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("loss_kd", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("loss_jsd",0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs) 
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # externel mt
        loss_sum_mt_ext = sum(log.get("loss_mt_ext", 0) for log in logging_outputs)
        nll_loss_sum_mt_ext = sum(log.get("nll_loss_mt_ext", 0) for log in logging_outputs)
        ntokens_mt_ext = sum(log.get("ntokens_mt_ext", 0) for log in logging_outputs)
        nsentences_mt_ext = sum(log.get("nsentences_mt_ext", 0) for log in logging_outputs) 
        sample_size_mt_ext = sum(log.get("sample_size_mt_ext", 0) for log in logging_outputs)

        


        # codebook ppl
        code_ppl_sum = sum(log.get("code_perplexity", 0) for log in logging_outputs)
        prob_ppl_sum = sum(log.get("prob_perplexity", 0) for log in logging_outputs)
        if sample_size > 0:
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "loss_mt", mt_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "loss_kd", kd_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "loss_jsd", jsd_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if nsentences > 0:
            metrics.log_scalar(
                "ctr_loss", ctr_loss_sum / nsentences, sample_size, round=3
            )
        if ntokens > 0:
            metrics.log_scalar(
                "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_scalar(
                "nll_loss_mt", nll_loss_mt_sum / ntokens / math.log(2), ntokens, round=3
            )

            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
            metrics.log_scalar(
            "code_ppl", code_ppl_sum / ntokens, ntokens, round=3
            )

            metrics.log_scalar(
                "prob_ppl", prob_ppl_sum / ntokens, ntokens, round=3
            )
        
        if sample_size_mt_ext > 0:
            metrics.log_scalar(
                "loss_mt_ext", loss_sum_mt_ext / sample_size_mt_ext / math.log(2), sample_size_mt_ext, round=3
            )
        if ntokens_mt_ext > 0:
            metrics.log_scalar(
                "nll_loss_mt_ext", nll_loss_sum_mt_ext / ntokens_mt_ext / math.log(2), ntokens_mt_ext, round=3
            )

            metrics.log_derived(
                "ppl_mt_ext", lambda meters: utils.get_perplexity(meters["nll_loss_mt_ext"].avg)
            )
        
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
