
set -e
DATA=$1
src=$2
tgt=$3
TOOLS_PATH=/path/to/tools
SPM_PATH=$TOOLS_PATH/sentencepiece/build/src
SPM_TRAIN=$SPM_PATH/spm_train
SPM_ENCODE=$SPM_PATH/spm_encode
NORM_PUNC=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
REPLACE_UNICODE_PUNCT=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl
REM_NON_PRINT_CHAR=$TOOLS_PATH/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl

CODES=32000

for split in "train" "valid"; do
    for lang in "$src" "$tgt"; do
        PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lang | $REM_NON_PRINT_CHAR"
        if [ ! -f "$DATA/$split.$lang.norm" ]; then
            eval "cat $DATA/$split.$lang | $PREPROCESSING > $DATA/$split.$lang.norm"
        fi
        
        # we tokenize chinese sentences by char.
        if [ $lang != "zh" ]; then
            if [ $split == "train" ]; then
                # train sp model
                mkdir -p $DATA/spm_models
                echo "$split split: train spm model for $lang"
                $SPM_TRAIN \
                    --input=$DATA/train.$lang.norm \
                    --model_prefix=$DATA/spm_models/$lang \
                    --vocab_size=$CODES \
                    --character_coverage=1.0 \
                    --model_type=bpe
            fi
            echo "$split split: spm encode for $lang"
            $SPM_ENCODE \
            --model $DATA/spm_models/$lang.model \
            --output_format=piece \
            --input $DATA/$split.$lang.norm \
            --output $DATA/$split.spm.$lang
        else
            echo "$split split: char tokenize for $lang"
            python `dirname $0`/char_tokenize.py $DATA/$split.$lang.norm $DATA/$split.spm.$lang
        fi
    done

    pref="--${split}pref $DATA/$split.spm"
    dict_config="--srcdict $DATA/dict.$src.txt --tgtdict $DATA/dict.$tgt.txt"
    config="--source-lang "$src" --target-lang "$tgt" \
            $pref \
            --destdir $DATA \
            --workers 12"
    if [ $split != "train" ]; then
        config=$config" $dict_config"
    fi

    echo "$split split: fairseq-preprocess config: $config"
    fairseq-preprocess $config
        
done
