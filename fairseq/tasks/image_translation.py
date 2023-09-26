# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II
import torch
import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.data import RawImageDataset,Image2TextDataset,FastRawImageDataset
from fairseq.utils import csv_str_list
import time
EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

def sampler(x, prob):
    # Sample from uniform distribution
    return np.random.choice(x, 1, p=prob).item()

@dataclass
class TranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    externel_data: Optional[str] = field(
        default="",
        metadata={
            "help": "path of externel mt data"
        },
    )
    task_prob: Optional[str] = field(
        default="0.9,0.1",
        metadata={
            "help": "a str of comma sperated weight for it task and externel mt task"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max pixel number in input image"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    # training config
    model_height: int = field (
        default=64,
        metadata={
            "help": "fix height for input image"
        },
    )
    model_width: int = field (
        default=600,
        metadata={
            "help": "fix width for input image"
        },
    )
    use_contrastive_learning: bool = field(
        default=False,
        metadata={
            "help": "use sentence-level contrastive learning between mt and it"
        },
    )

    ctr_lamda: float = field(
        default=1.0,
        metadata={"help": "weight for contrastive learning"},
    )

    teacher_mt_dir: str = field(
        default="",
        metadata={"help": "text machine translation model directory for contrastive learning or knowledge distillation"},
    )

    teacher_mt_data_dir: str = field(
        default="",
        metadata={"help": "data directory of teacher_mt_dir"},
    )


    use_knowledge_distillation: bool = field(
        default=False,
        metadata={
            "help": "use knowledge distillation from mt to it"
        },
    )

    kd_lamda: float = field(
        default=0.0,
        metadata={"help": "weight for knowledge distillation"},
    ) 

    use_multi_task_learning: bool = field(
        default=False,
        metadata={
            "help": "use multi-task learning including mt and it translation"
        },
    )

    mtl_it_lamda: float = field(
        default=1.0,
        metadata={"help": "weight for image-translation in multi-task learning"},
    )

    mtl_mt_lamda: float = field(
        default=0.0,
        metadata={"help": "weight for text-translation in multi-task learning"},
    )

    use_jsd: bool = field(
        default=False,
        metadata={
            "help": "use jsd between mt and it"
        },
    )

    jsd_lamda: float = field(
        default=0.0,
        metadata={
            "help": "weight for jsd"
        },
    )

    pipeline: bool = field(
        default=False,
        metadata={"help": "use pipeline(OCR+MT) in inference step"},
    )
    pipeline_src_spm: str = field(
        default='',
        metadata={"help": "path of source sentencepiece model to tokenize ocr result"},
    )



@register_task("image_translation", dataclass=TranslationConfig)
class Image_TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_path = self.cfg.data
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        # infer langcode
        # src, tgt = self.cfg.source_lang, self.cfg.target_lang

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))

        
        self.datasets[split] = RawImageDataset(
            manifest_path=manifest_path,
            model_height=self.cfg.model_height,
            model_width=self.cfg.model_width,
            min_sample_size=self.cfg.min_sample_size,
            text_compression_level=text_compression_level,
        )
        src = self.cfg.source_lang
        tgt = self.cfg.target_lang
        prefix = f"{data_path}/{split}.{src}-{tgt}."
        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, self.src_dict, self.cfg.dataset_impl
        )

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, self.tgt_dict, self.cfg.dataset_impl
        )
        src_sizes, tgt_sizes = src_dataset.sizes, tgt_dataset.sizes
        self.datasets[split] = Image2TextDataset(
            self.datasets[split],
            src_dataset,
            tgt_dataset,
            src_sizes,
            tgt_sizes,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
        )

        # loading externel mt data
        ext_mt_dataset = None
        if split == self.cfg.train_subset and self.cfg.externel_data:
            externel_data_path = self.cfg.externel_data
            src, tgt = self.cfg.source_lang, self.cfg.target_lang
            from .translation import load_langpair_dataset

            ext_mt_dataset = load_langpair_dataset(
                        externel_data_path,
                        split,
                        src,
                        self.src_dict,
                        tgt,
                        self.tgt_dict,
                        combine=combine,
                        dataset_impl=self.cfg.dataset_impl,
                        upsample_primary=self.cfg.upsample_primary,
                        left_pad_source=self.cfg.left_pad_source,
                        left_pad_target=self.cfg.left_pad_target,
                        max_source_positions=self.cfg.max_source_positions,
                        max_target_positions=self.cfg.max_target_positions,
                        load_alignments=self.cfg.load_alignments,
                        truncate_source=self.cfg.truncate_source,
                        num_buckets=self.cfg.num_batch_buckets,
                        shuffle=True,
                        pad_to_multiple=self.cfg.required_seq_len_multiple,
                    )
            # construct RoundRobinZipDatasets
            from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets
            from collections import OrderedDict
            it_dataset = self.datasets[split]
            mt_dataset = ext_mt_dataset
            datasets = OrderedDict([("it", it_dataset), ("mt", mt_dataset)])
            self.datasets[split] = RoundRobinZipDatasets(datasets)
            # construct multi-corpus dataset

            # from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
            # from collections import OrderedDict
            # from functools import partial

            # it_dataset = self.datasets[split]

            # multi_corpus = OrderedDict([("it", it_dataset), ("mt", mt_dataset)])
            # task_prob = [float(x) for x in self.cfg.task_prob.split(",")]
            
            
            # self.datasets[split] = MultiCorpusSampledDataset(multi_corpus, partial(sampler, prob=task_prob))
            
        



    def build_dataset_for_inference(self, img_paths, src_lengths, constraints=None):
        return FastRawImageDataset(
            img_paths,
            model_height=self.cfg.model_height,
            model_width=self.cfg.model_width,
        )

    def build_model(self, cfg, from_checkpoint=False):
        
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        self.text_model = None
        if self.cfg.teacher_mt_dir and self.cfg.teacher_mt_data_dir:
            logger.info("loading pretrained text mt model")
            # load pretrained text mt model
            from fairseq.models.transformer.transformer_legacy import TransformerModel
            self.text_model = TransformerModel.from_pretrained(model_name_or_path=self.cfg.teacher_mt_dir,
                                                               data_name_or_path=self.cfg.teacher_mt_data_dir)
            self.text_model.eval()
        return model
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if self.cfg.pipeline:
            # load easyocr model
            import easyocr
            self.reader = easyocr.Reader(['ch_sim','en'],gpu=True,detector=False,recognizer=True,quantize=False)
            # load cnocr model
            # from cnocr import CnOcr
            # self.cnocr = CnOcr(rec_model_name='densenet_lite_134-gru', det_model_name=None, context="gpu")  # 所有参数都使用默认值
            # # calculate parameter number
            # sum_p = sum(
            #     p.numel() for p in self.cnocr.rec_model._model.parameters()
            # )
            # logger.info("Parameters of cnocr model: ", sum_p)
            
            
            
            # load sentencepiece model for source
            if self.cfg.pipeline_src_spm:
                import sentencepiece as spm
                self.src_spm = spm.SentencePieceProcessor()
                self.src_spm.load(self.cfg.pipeline_src_spm)
            else:
                self.src_spm = None

        return super().build_generator(models,
                                        args,
                                        seq_gen_cls=seq_gen_cls,
                                        extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,)
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):

        if self.cfg.pipeline:
            device = sample["id"].device
            # easyocr
            
            
            from easyocr.recognition import recognizer_predict
            star = time.time()
            batch_max_length = int(self.cfg.model_width/10) # 10 pixel for a single prediction
            ignore_idx = []
            char_group_idx = {}
            ocr_result, pred_time, decode_time = recognizer_predict(self.reader.recognizer,
                                self.reader.converter,
                                [sample["net_input"]["img_source"]],
                                batch_max_length,
                                ignore_idx,
                                char_group_idx)
            src_string = [item[0] for item in ocr_result]
            ocr_time = time.time() - star
            
            # cnocr
            # dataset = self.datasets["test"].dataset
            
            # img_paths = [os.path.join(dataset.root_dir, dataset.fnames[idx]) for idx in sample["id"]]

            # ocr_result = self.cnocr.ocr_for_single_lines(img_paths, batch_size=len(img_paths))
            # src_string = [item["text"] for item in ocr_result]

            # spm
            process_star = time.time()
            if self.src_spm is not None:
                src_string = self.src_spm.encode(src_string, out_type=str)
            else:
                src_string = [list(s) for s in src_string] # char split
            src_string = [' '.join(s) for s in src_string]
            # encode
            # norm punc
            from fairseq.data.image.image_utils import norm_punc
            sequences = [self.source_dictionary.encode_line(norm_punc(s), add_if_not_exist=False) for s in src_string]
            # collater
            from fairseq.data import data_utils
            src_dict = self.source_dictionary
            src_tokens = data_utils.collate_tokens(sequences, pad_idx=src_dict.pad(), left_pad=True).to(device)

            mt_sample = {}
            mt_sample["id"] = sample["id"]
            mt_sample["net_input"] = {"src_tokens": src_tokens, "src_lengths":torch.LongTensor([len(t) for t in sequences]).to(device)}
            process_time = time.time() - process_star
            with torch.no_grad():
                
                mt_star = time.time()
                res = generator.generate(
                    models, mt_sample, prefix_tokens=prefix_tokens, constraints=constraints
                )
                mt_time = time.time() - mt_star
                logger.info(f"ocr time: {ocr_time}, pred_time: {pred_time}, decode_time:{decode_time}, process_time: {process_time}, mt_time: {mt_time} ")
                return res
        else:
            with torch.no_grad():
                it_star = time.time()
                res = generator.generate(
                    models, sample, prefix_tokens=prefix_tokens, constraints=constraints
                )
                it_time = time.time() - it_star
                logger.info(f"it_time: {it_time}")
                return res


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.metrics.BLEU().corpus_score(hyps, [refs])
        else:
            return sacrebleu.metrics.BLEU(trg_lang=self.cfg.target_lang).corpus_score(
                hyps, [refs]
            )

    def get_interactive_tokens_and_lengths(self, img_paths, encode_fn):
        lengths = None
        return img_paths, lengths

