#!/usr/bin/env python3
import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images_dir", metavar="DIR", help="root directory containing image files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--dataset-prefix", default="train", type=str, help="prefix of dataset"
    )
    parser.add_argument(
        "--ext", default="jpg", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--target-cnt", default=10000, type=int, help="the size of dataset"
    )
    parser.add_argument(
        "--langs", type=str, help="langs of extracted text"
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    langs = args.langs.split(',')
    assert len(langs) > 0, "Need to specify langs"
    for lang in langs:
        assert os.path.exists(f"{args.images_dir}/text.{lang}")
    text_readers = [open(f"{args.images_dir}/text.{lang}", "r") for lang in langs]
    text_writers = [open(f"{args.dest}/{args.dataset_prefix}.{lang}", "w") for lang in langs]
    with open(f"{args.dest}/{args.dataset_prefix}.tsv", "w") as tsv:
        tsv.write(args.images_dir.strip() + "\n")
        for i in range(args.target_cnt):
            if not os.path.exists(f"{args.images_dir}/{i}.{args.ext}"):
                break
            
            tsv.write(f"{i}.{args.ext}\t0" + "\n")
            text_lines = [reader.readline() for reader in text_readers]

            for line, writer in zip(text_lines, text_writers):
                writer.write(line.strip() + "\n")
        



