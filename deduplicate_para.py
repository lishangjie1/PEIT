#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput
import hashlib
import sys
import string
from multiprocessing import Pool
from zhon.hanzi import punctuation 


english_punc = string.punctuation
chinese_punc = punctuation
punc = english_punc + chinese_punc
def get_hashes_and_lines(raw_line):
    if isinstance(raw_line, str):
        new_line = "".join([c for c in raw_line if c not in punc])
        hash = hashlib.md5(new_line.encode().strip()).hexdigest()
        return hash, raw_line
    elif isinstance(raw_line, tuple):
        src, tgt = raw_line[0], raw_line[1]
        new_line = "".join([c for c in src if c not in punc] + [c for c in tgt if c not in punc])
        hash_src = hashlib.md5(new_line.encode().strip()).hexdigest()
        #hash_tgt = hashlib.md5(raw_line[1].strip()).hexdigest()
        return hash_src, raw_line
    else:
        raise Exception("wrong input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("rec_files_src", help="recent input source files")
    parser.add_argument("rec_files_tgt", help="recent input source files")
    parser.add_argument("out_file_src", help="recent output source files")
    parser.add_argument("out_file_tgt", help="recent output source files")
    args = parser.parse_args()

    seen = set()
    with open(args.rec_files_src, mode="r") as h_src, open(args.rec_files_tgt, mode="r") as h_tgt, \
        open(args.out_file_src, mode="w") as w_src, open(args.out_file_tgt, mode="w") as w_tgt:
        pool = Pool(args.workers)
        results = pool.imap_unordered(get_hashes_and_lines, zip(h_src,h_tgt), 1000)
        for i, (hash, raw_line) in enumerate(results):
            if hash not in seen:
                seen.add(hash)
                w_src.write(raw_line[0].strip() + "\n")
                w_tgt.write(raw_line[1].strip() + "\n")

            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
    
    print(file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
