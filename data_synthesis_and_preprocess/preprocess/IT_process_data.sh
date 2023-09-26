
set -e
IT_DATA=$1
MT_DATA=$2
TOOLS_PATH=/path/to/tools
src=$3
tgt=$4

# image manifest
train_images_path="$IT_DATA/train_images"
valid_images_path="$IT_DATA/valid_images"
dest="$IT_DATA/dest"
ext="jpg"
langs="$src,$tgt"

# create train dataset
python `dirname $0`/image_manifest.py $train_images_path \
    --dest $dest \
    --dataset-prefix "train" \
    --ext $ext \
    --langs $langs \
    --target-cnt 10000000

#create valid dataset
python `dirname $0`/image_manifest.py $valid_images_path \
    --dest $dest \
    --dataset-prefix "valid" \
    --ext $ext \
    --langs $langs \
    --target-cnt 5000




SPM_PATH=$TOOLS_PATH/sentencepiece/build/src
SPM_TRAIN=$SPM_PATH/spm_train
SPM_ENCODE=$SPM_PATH/spm_encode
NORM_PUNC=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
REPLACE_UNICODE_PUNCT=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
REM_NON_PRINT_CHAR=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl

SRC_DICT=$MT_DATA/dict.$src.txt # source dictionary from Text MT
TGT_DICT=$MT_DATA/dict.$tgt.txt # target dictionary from Text MT

for split in "train" "valid"; do
    echo "Preprocessing $split split..."
    for lang in "$src" "$tgt"; do
        PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lang | $REM_NON_PRINT_CHAR"
        eval "cat $dest/$split.$lang | $PREPROCESSING > $dest/$split.$lang.norm"
        # we tokenize chinese sentences by char.
        if [ $lang != "zh" ]; then 
            # SentencePiece model from from Text MT 
            $SPM_ENCODE \
            --model $MT_DATA/spm_models/$lang.model \
            --output_format=piece \
            --input $dest/$split.$lang.norm \
            --output $dest/$split.spm.$lang
        else
            python `dirname $0`/char_tokenize.py $dest/$split.$lang.norm $dest/$split.spm.$lang
        fi
    done

    pref="--${split}pref $dest/$split.spm"
    config="--source-lang "$src" --target-lang "$tgt" \
            $pref \
            --destdir $dest \
            --workers 12 \
            --srcdict $SRC_DICT \
            --tgtdict $TGT_DICT"
    fairseq-preprocess $config
done
