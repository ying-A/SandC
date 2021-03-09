#!/bin/bash

# train the customzie model without skeletons augmentation
python customize_train.py --unique_flag without_aug\
    --evaluate_during_training

# train the customzie model with skeletons augmentation
python customize_train.py --unique_flag with_aug\
    --evaluate_during_training\
    --train_raw_path data/aug_merge_train_skeletons_supervised_large.json\
    --train_tokenized_path data/aug_merge_train_tokenized_supervised_large.txt\
    --train_lens_path data/aug_merge_train_lens_supervised_large.txt