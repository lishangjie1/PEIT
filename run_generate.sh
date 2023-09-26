#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
src="zh"
tgt="en"
tmp=/path/to/tmp
MT_DATA=$tmp/data/mt_data
# spm
SPM_MODEL=$MT_DATA/spm_models/$tgt.model
# test data path
test_path="/$tmp/data/ft_data/dest"
# model dir
model_dir=$tmp/models/ft_it_model_zh_to_en
# output dir
generate_dir="$tmp/generate_dir"
mkdir -p $generate_dir
pretrained_ocr_path="/path/to/ocr/zh_sim_g2.pth"

split="test"
input_path="$test_path"
task="image_translation"
fairseq-generate "$input_path" \
                --path "$model_dir/checkpoint_best.pt"  \
                --task  "$task"  \
                --gen-subset  "$split" \
                --beam 5 \
                -s "zh" -t "en" \
                --bpe  "sentencepiece"  --sentencepiece-model  "$SPM_MODEL" \
                --scoring "sacrebleu" \
                --batch-size  "128" \
                --results-path "$generate_dir" \
                --skip-invalid-size-inputs-valid-test \
                --model-overrides "{'pretrained_ocr_path':'$pretrained_ocr_path'}"

cat $generate_dir/generate-$split.txt | tail -1

cat $generate_dir/generate-$split.txt | grep -P "^D" | sort -V | cut -f 3- > $generate_dir/hyp
cat $generate_dir/generate-$split.txt | grep -P "^T" | sort -V | cut -f 2- > $generate_dir/ref

echo "====BLEU score is ...."
cat $generate_dir/hyp | sacrebleu -b $generate_dir/ref
