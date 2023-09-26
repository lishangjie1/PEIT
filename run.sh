
# example: translate from zh to en


# initial directory structure
# train_images and valid_images can be created by peit/data_synthesis_and_preprocess/data_synthesis/data_synthesis.py
# /tmp/data
# |—— mt_data (parallel text data)
#     |—— train.en
#     |—— train.zh
#     |—— valid.en
#     |—— valid.zh

# |—— it_data (Synthetic Data)
#     |—— train_images
#         |—— 0.jpg
#         |—— 1.jpg
#         |—— ....
#         |—— 10000000.jpg
#         |—— text.en
#         |—— text.zh
#     |—— valid_images
#         |—— 0.jpg
#         |—— 1.jpg
#         |—— ....
#         |—— 1000.jpg
#         |—— text.en
#         |—— text.zh
# |—— ft_data (ECOIT)
#     |—— train_images
#         |—— 0.jpg
#         |—— 1.jpg
#         |—— ....
#         |—— 470000.jpg
#         |—— text.en
#         |—— text.zh
#     |—— valid_images
#         |—— 0.jpg
#         |—— 1.jpg
#         |—— ....
#         |—— 1000.jpg
#         |—— text.en
#         |—— text.zh


src="zh"
tgt="en"
# ================= stage 1: MT Pre-Training ==================
MAIN=/path/to/peit
tmp=/path/to/tmp
bash $MAIN/data_synthesis_and_preprocess/preprocess/MT_process_data.sh $tmp/data/mt_data $src $tgt

# training 
max_update=10
text_dest="$tmp/data/mt_data"
text_model_path="$tmp/models/mt_model_${src}_to_${tgt}"
mkdir -p $text_model_path

CUDA_VISIBLE_DEVICES=0 fairseq-train $text_dest \
--save-dir $text_model_path --source-lang "zh" --target-lang "en" --fp16 --max-update $max_update --save-interval-updates 4000 --keep-interval-updates 1 --no-epoch-checkpoints \
--arch transformer --task translation --optimizer adam --adam-betas '(0.9, 0.98)' --dropout 0.1 --lr 1e-4 --lr-scheduler inverse_sqrt \
--log-interval 10 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 12000 --update-freq 2 --skip-invalid-size-inputs-valid-test \
--encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 8 --decoder-attention-heads 8 --share-decoder-input-output-embed --layernorm-embedding \
--dataset-impl "mmap" --ddp-backend "c10d" 

# ================= stage 2: IT Pre-Training ==================


bash $MAIN/data_synthesis_and_preprocess/preprocess/IT_process_data.sh $tmp/data/it_data $tmp/data/mt_data  $src $tgt

max_update=100000
text_data=$text_dest
text_model_dir=$text_model_path
it_dest="$tmp/data/it_data/dest"
it_model_path="$tmp/models/it_model_${src}_to_${tgt}"
pretrained_ocr_path=/path/to/ocr/zh_sim_g2.pth


CUDA_VISIBLE_DEVICES=0 fairseq-train $it_dest \
-s "zh" -t "en" \
--save-dir $it_model_path \
--restore-file "$text_model_path/checkpoint_best.pt" \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
--fp16 --max-update $max_update --save-interval-updates 5000 --no-epoch-checkpoints --keep-interval-updates 1 \
--arch itransformer --task image_translation --optimizer adam --adam-betas "(0.9, 0.98)" --lr 1e-4 --end-learning-rate 1e-5 --lr-scheduler polynomial_decay --total-num-update $max_update \
--criterion label_smoothed_cross_entropy_contrastive --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 \
--batch-size 300 --max-tokens 4096 --update-freq "1" --skip-invalid-size-inputs-valid-test --log-interval 1 \
--dataset-impl "mmap" --ddp-backend=legacy_ddp \
--encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--num-workers 2 \
--use-multi-task-learning --mtl-it-lamda 1.0 --mtl-mt-lamda 1.0 \
--use-contrastive-learning --ctr-lamda 1.0 \
--teacher-mt-dir "$text_model_dir" \
--teacher-mt-data-dir "$text_data" \
--use-knowledge-distillation --kd-lamda 1.0 \
--use-jsd --jsd-lamda 1.0 \
--use-pretrained-ocr \
--pretrained-ocr-path "$pretrained_ocr_path"

# --multi-line --model-height 320 --model-width 480 \
# ================= stage 3: IT Fine-Tuning ==================

bash $MAIN/data_synthesis_and_preprocess/preprocess/IT_process_data.sh $tmp/data/ft_data $tmp/data/mt_data  $src $tgt
IMAGE_DATA=$tmp/data/ft_data
dest="$IMAGE_DATA/dest"

it_pretrained_model_path="$it_model_path/checkpoint_best.pt"
ft_model_path="$tmp/models/ft_it_model_${src}_to_${tgt}"
max_updates=30000
CUDA_VISIBLE_DEVICES=0 fairseq-train $dest \
-s "zh" -t "en" \
--save-dir $ft_model_path \
--restore-file "$it_pretrained_model_path" \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
--fp16 --max-update $max_updates --save-interval-updates 5000 --no-epoch-checkpoints --keep-interval-updates 1 \
--arch itransformer --task image_translation --optimizer adam --adam-betas "(0.9, 0.98)" --lr 3e-5 --end-learning-rate 1e-5 --lr-scheduler polynomial_decay --total-num-update $max_updates \
--criterion label_smoothed_cross_entropy_contrastive --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 \
--batch-size 200 --max-tokens 1024 --update-freq "1" --skip-invalid-size-inputs-valid-test --log-interval 1 \
--dataset-impl "mmap" --ddp-backend=legacy_ddp \
--encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--eval-bleu --eval-bleu-remove-bpe "sentencepiece" \
--use-pretrained-ocr \

# --multi-line --model-height 320 --model-width 480 \
