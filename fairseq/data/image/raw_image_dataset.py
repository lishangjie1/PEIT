
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import io
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset,data_utils
from fairseq.data.image.image_utils import (
    parse_path,
    resize_aspect_ratio,
    normalizeMeanVariance,
    reformat_input,
    compute_ratio_and_resize
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
import cv2
import torchvision.transforms as transforms
import math
import time
logger = logging.getLogger(__name__)

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low


def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :h, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :h, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        # if self.max_size[1] != h: # bottom pad
        #     Pad_img[:, h:, :] = Pad_img[:, h - 1 , :].unsqueeze(1).expand(c, self.max_size[1] - h, self.max_size[2])
        return Pad_img



class RawImageDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path,
        model_height=64,
        model_width=600,
        min_sample_size=0,
        adjust_contrast=0.0,
        shuffle=True,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.adjust_contrast = adjust_contrast
        self.shuffle = shuffle
        self.text_compressor = TextCompressor(level=text_compression_level)
        self.model_height = model_height
        self.model_width = model_width
        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()
        self.manifest_path = manifest_path
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(items[0])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        #fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path = parse_path(path_or_fp)
        img = cv2.imread(_path, flags=0)
        img = self.preprocess(img)
        img = self.postprocess(img)
        return {"id": index, "img_source": img}

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)

    def preprocess(self, img_grey):
        # preprocess for recognizer
        #img, img_grey = reformat_input(img) # obtain grey image
        height, width = img_grey.shape
        resized_img = compute_ratio_and_resize(img_grey, 
                                                     width, 
                                                     height, 
                                                     self.model_width, 
                                                     self.model_height) # fix height as model_height
        return Image.fromarray(resized_img, 'L')

    def collater(self, samples):
        samples = [s for s in samples if s["img_source"] is not None]
        if len(samples) == 0:
            return {}

        # obtain maximum height and width, then pad
        # max_width = max([sample["img_source"].size[0] for sample in samples])
        input_channel = 1
        transform = NormalizePAD((input_channel, self.model_height, self.model_width))
        transformed_images = []
        for sample in samples:
            image = sample["img_source"]
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L")) # 256 grey
                image = adjust_contrast_grey(image, target = self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            transformed_images.append(transform(image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in transformed_images], 0) # (B,C,H,W)
        input = {"img_source": image_tensors}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        out["net_input"] = input
        return out
    
    def postprocess(self, feats):
        # todo image postprocess
        return feats




class Image2TextDataset(FairseqDataset):

    def __init__(
        self,
        dataset,
        src_labels,
        tgt_labels,
        src_sizes,
        tgt_sizes,
        pad,
        eos,
        ext_mt_dataset=None
    ):
        super().__init__()
        self.dataset = dataset
        self.src_labels = src_labels
        self.tgt_labels = tgt_labels
        self.pad = pad
        self.eos = eos

        
        self.src_sizes = src_sizes
        assert len(self.src_sizes) == len(self.dataset)

        self.tgt_sizes = tgt_sizes
        assert len(self.tgt_sizes) == len(self.dataset)
        
    def __getitem__(self, index):
        item = self.dataset[index]
        if self.src_labels is not None:
            item["src_label"] = self.src_labels[index]
        item["tgt_label"] = self.tgt_labels[index]
        return item
    
    def __len__(self):
        return len(self.src_sizes)


    def size(self, index):
        src_size = self.src_sizes[index]
        tgt_size = self.tgt_sizes[index]
        return src_size, tgt_size

    def num_tokens(self, index):
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index],
        )
    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes
    def collater(self, samples):

        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        source = [s["src_label"] for s in samples if s["id"] in indices] if self.src_labels is not None else None
        target = [s["tgt_label"] for s in samples if s["id"] in indices]


        collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
        collated["net_input"]["source_lengths"] = torch.LongTensor([len(t) for t in source]) if source is not None else None

        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        source = data_utils.collate_tokens(source, pad_idx=self.pad, left_pad=True) if source is not None else None

        collated["ntokens"] = collated["target_lengths"].sum().item()
        collated["net_input"]["prev_output_tokens"] = data_utils.collate_tokens(
                                            [s['tgt_label'] for s in samples],
                                            self.pad,
                                            self.eos,
                                            left_pad=False,
                                            move_eos_to_beginning=True,
                                        )

        collated["target"] = target
        collated["net_input"]["source_text"] = source
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
        return indices, ignored
    def ordered_indices(self, shuffle=True):
        
        if shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        # sort by target length, then source length
        # indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        #indices = indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        return indices




class FastRawImageDataset(FairseqDataset):
    def __init__(
        self,
        img_paths,
        model_height=64,
        model_width=600,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__()
        self.text_compressor = TextCompressor(level=text_compression_level)
        self.model_height = model_height
        self.model_width = model_width
        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        for fname in img_paths:
            self.fnames.append(self.text_compressor.compress(fname))
            sizes.append(1)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):
        
        _path = self.fnames[index]
        _path = self.text_compressor.decompress(_path)

        img = cv2.imread(_path)
        img = self.preprocess(img)
        #feats = torch.from_numpy(img).float()
        img = self.postprocess(img)
        return {"id": index, "img_source": img}

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)

    def preprocess(self, img):
        # preprocess for recognizer
        img, img_grey = reformat_input(img) # obtain grey image
        height, width = img_grey.shape
        resized_img = compute_ratio_and_resize(img_grey,width,height,self.model_width,self.model_height) # fix height as model_height
        return Image.fromarray(resized_img, 'L')

    def collater(self, samples):
        samples = [s for s in samples if s["img_source"] is not None]
        if len(samples) == 0:
            return {}
        # obtain maximum height and width, then pad
        #max_width = max([sample["img_source"].size[0] for sample in samples])
        input_channel = 1
        transform = NormalizePAD((input_channel, self.model_height, self.model_width))
        transformed_images = []
        for sample in samples:
            img = sample["img_source"]
            w, h = img.size
            transformed_images.append(transform(img))

        image_tensors = torch.cat([t.unsqueeze(0) for t in transformed_images], 0) # (B,C,H,W)
        input = {"img_source": image_tensors}
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        out["net_input"] = input
        return out
    
    def postprocess(self, feats):
        # todo image postprocess
        return feats