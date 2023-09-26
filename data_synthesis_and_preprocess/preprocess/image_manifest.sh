
set -e
tmp=/data/lsj/tmp
IT_DATA=$tmp/data/it_data
train_images_path="$IT_DATA/train_images"
valid_images_path="$IT_DATA/valid_images"
dest="$IT_DATA/dest"
ext="jpg"
langs="en,zh"
# create train dataset
python image_manifest.py $train_images_path \
    --dest $dest \
    --dataset-prefix "train" \
    --ext $ext \
    --langs $langs \
    --target-cnt 100

#create valid dataset
python image_manifest.py $valid_images_path \
    --dest $dest \
    --dataset-prefix "valid" \
    --ext $ext \
    --langs $langs \
    --target-cnt 10




